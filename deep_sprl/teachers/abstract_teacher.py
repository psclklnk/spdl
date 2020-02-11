import gym
import numpy as np
from abc import ABC, abstractmethod
from deep_sprl.teachers.util import Buffer


class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self):
        pass


class BaseWrapper(gym.Env):

    def __init__(self, env, teacher, discount_factor, context_visible):
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

        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.context_visible = context_visible
        self.cur_context = None
        self.cur_initial_state = None

    def done_callback(self, step):
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
        self.undiscounted_reward += step[1]
        self.discounted_reward += self.cur_disc * step[1]
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if step[2]:
            self.done_callback(step)

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
