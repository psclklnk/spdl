import gym
import numpy as np
from gym import spaces
from abc import ABC, abstractmethod
from deep_sprl.teachers.util import Buffer
from stable_baselines.common.vec_env import VecEnv


class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self):
        pass


class BaseWrapper(gym.Env):

    def __init__(self, env, teacher, discount_factor, context_visible, reward_from_info=False):
        gym.Env.__init__(self)
        self.stats_buffer = Buffer(3, 1000, True)

        self.env = env
        self.teacher = teacher
        self.discount_factor = discount_factor

        if context_visible:
            context = self.teacher.sample()
            low_ext = np.concatenate((self.env.observation_space.low, -np.inf * np.ones_like(context)))
            high_ext = np.concatenate((self.env.observation_space.high, np.inf * np.ones_like(context)))
            self.observation_space = gym.spaces.Box(low=low_ext, high=high_ext)
        else:
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.context_visible = context_visible
        self.cur_context = None
        self.cur_initial_state = None

        self.reward_from_info = reward_from_info

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        pass

    def step(self, action):
        step = self.env.step(action)
        if self.context_visible:
            step = np.concatenate((step[0], self.cur_context)), step[1], step[2], step[3]
        self.update(step)
        return step

    def reset(self):
        self.cur_context = self.teacher.sample()
        self.env.unwrapped.context = self.cur_context.copy()
        obs = self.env.reset()

        if self.context_visible:
            obs = np.concatenate((obs, self.cur_context))

        self.cur_initial_state = obs.copy()
        return obs

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def update(self, step):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if step[2]:
            self.done_callback(step, self.cur_initial_state.copy(), self.cur_context.copy(), self.discounted_reward)

            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, self.step_length))
            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.cur_initial_state = None

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_step_length = np.mean(steps)

            return mean_reward, mean_disc_reward, mean_step_length


class BaseIsaacWrapper(VecEnv):

    def __init__(self, create_exp, teacher, discount_factor, context_visible, view=False, sync_frame_time=False,
                 num_envs=40):
        self.stats_buffer = Buffer(3, 1000, True)
        self.discount_factor = discount_factor
        self.undiscounted_rewards = np.zeros(num_envs)
        self.discounted_rewards = np.zeros(num_envs)
        self.cur_discs = np.ones(num_envs)
        self.step_lengths = np.zeros(num_envs)

        self.gym, self.sim, self.exp, self.viewer = create_exp(num_envs, teacher, view)
        self.context_visible = context_visible

        self.sync_frame_time = sync_frame_time

        self.num_envs = num_envs
        self.cur_actions = None

        self.teacher = teacher
        if context_visible:
            context = self.teacher.sample()
            low_ext = np.concatenate((np.ones(self.exp._num_obs) * -np.Inf, -np.inf * np.ones_like(context)))
            high_ext = np.concatenate((np.ones(self.exp._num_obs) * np.Inf, np.inf * np.ones_like(context)))
            observation_space = spaces.Box(low_ext, high_ext)
        else:
            observation_space = spaces.Box(np.ones(self.exp._num_obs) * -np.Inf, np.ones(self.exp._num_obs) * np.Inf)
        action_space = spaces.Box(np.ones(self.exp._num_actions) * -1., np.ones(self.exp._num_actions) * 1.)
        super().__init__(num_envs, observation_space, action_space)

    def step(self, actions):
        step = self.exp.step(actions)
        if self.context_visible:
            step = np.concatenate((step[0], self.exp._task.cur_contexts), axis=1), step[1], step[2], step[3]
        self.update(step)
        self.render()
        return step

    def reset(self, reset_all=False):
        obs = self.exp.reset(reset_all=reset_all)
        if self.context_visible:
            return np.concatenate((obs, self.exp._task.cur_contexts), axis=1)
        else:
            return obs

    def get_obs_shape(self):
        return self.exp.get_obs_shape()

    def get_num_obs(self):
        return self.exp.get_num_obs()

    def get_num_actions(self):
        return self.exp.get_num_actions()

    def render(self):
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if self.sync_frame_time:
                self.gym.sync_frame_time(self.sim)

    def close(self):
        print("Close called")
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def env_method(self, name, *args, **kwargs):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, *method_args, indices=None, **method_kwargs):
        pass

    def step_async(self, actions):
        self.cur_actions = actions

    def step_wait(self):
        step = self.exp.step(self.cur_actions)
        if self.context_visible:
            step = np.concatenate((step[0], self.exp._task.cur_contexts), axis=1), step[1], step[2], step[3]
        self.update(step)

        self.render()
        self.cur_actions = None
        return step

    def get_images(self):
        return None

    def update(self, step):
        # Update the statistics
        self.undiscounted_rewards += step[1]
        self.discounted_rewards += self.cur_discs * step[1]
        self.cur_discs *= self.discount_factor
        self.step_lengths += 1.
        self.stats_buffer.update_buffer((self.undiscounted_rewards[step[2]].tolist(),
                                         self.discounted_rewards[step[2]].tolist(),
                                         self.step_lengths[step[2]].tolist()))

        if np.any(step[2]):
            for i in range(0, self.num_envs):
                if step[2][i]:
                    cur_context = self.exp._task.cur_contexts[i, :].copy()
                    self.done_callback((step[0][i, :], step[1][i], step[2][i], step[3][i]),
                                       np.concatenate((self.exp._task.cur_initial_states[i, :], cur_context)),
                                       cur_context, self.discounted_rewards[i])

            # Reset the environments that have finished running
            self.exp._task.reset(np.where(step[2].squeeze())[0].flatten())

        self.undiscounted_rewards[step[2]] = 0.
        self.discounted_rewards[step[2]] = 0.
        self.cur_discs[step[2]] = 1.
        self.step_lengths[step[2]] = 0.

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_step_length = np.mean(steps)

            return mean_reward, mean_disc_reward, mean_step_length

    def done_callback(self, step, cur_initial_state, cur_context, discount_reward):
        pass
