from __future__ import print_function, division, absolute_import

import abc
from math import sqrt
import numpy as np

from carbongym import gymapi


class Environment(abc.ABC):
    """ A base class for environment instances """

    def __init__(self, **kwargs):
        self._gym = kwargs["gym"]
        self._envPtr = kwargs["env"]
        self._envIndex = kwargs["env_index"]

    @abc.abstractmethod
    def num_actions(self):
        """ Number of actions that this environment accepts. """

    @abc.abstractmethod
    def get_null_action(self):
        """ returns the 'do nothing' action """

    @abc.abstractmethod
    def step(self, actions):
        """ Callback for stepping the environment.  this actually articulates the objects in the env """

    @staticmethod
    @abc.abstractmethod
    def create_shared_data(gym):
        """ Create or load assets to be shared by all environment instances """


class Task(abc.ABC):
    """ A base class for RL tasks to be performed in an Environment """

    def __init__(self, envs, **kwargs):
        self._gym = kwargs["gym"]
        self._envs = np.array(envs)
        self._envPtrs = np.array([env._envPtr for env in envs])
        self._numEnvs = len(envs)

    @abc.abstractmethod
    def num_observations(self):
        """ Number of observations that this task generates. """

    @abc.abstractmethod
    def fill_observations(self, actions):
        """ fill and return observation"""

    @abc.abstractmethod
    def fill_rewards(self, actions):
        """ fill and return rewards"""

    @abc.abstractmethod
    def fill_dones(self, actions):
        """ fill and return dones"""

    @abc.abstractmethod
    def reset(self, kills):
        """ Callback to re-initialize specific environments in the task """


class Experiment:

    def __init__(self, gym, sim, EnvClass, TaskClass, num_envs, spacing, viewer=None, env_param=None, task_param=None, asset_path=None):
        self._first_reset = True
        self._num_envs = num_envs
        self._gym = gym
        self._sim = sim

        # acquire data to be shared by all environment instances
        if asset_path is not None:
            shared_data = EnvClass.create_shared_data(self._gym, self._sim, data_dir=asset_path)
        else:
            shared_data = EnvClass.create_shared_data(self._gym, self._sim)

        # env bounds
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        if viewer is not None:
            self._viewer = viewer
        # create environment and task instances
        self.envs = []

        if env_param is None:
            env_param = dict()

        if task_param is None:
            task_param = dict()

        num_per_row = int(sqrt(num_envs))
        for i in range(num_envs):
            # create env instance
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)
            env_base_args = {
                "gym": self._gym,
                "env": env_ptr,
                "env_index": i,
            }
            env = EnvClass(shared_data, viewer=viewer, **env_param, **env_base_args)
            self.envs.append(env)

        task_base_args = {"gym": self._gym}
        self._task = TaskClass(self.envs, **task_param, **task_base_args)
        self._num_actions = self.envs[0].num_actions()  # this is a safe assumption as it is required for vectorized training of any env
        self._num_obs = self._task.num_observations()

        self.observation_space = np.array([self._num_obs, ])
        self.action_space = np.array([self._num_actions, ])

    def get_obs_shape(self):
        return [self._task.num_observations(), ]

    def get_num_obs(self):
        return self._num_obs

    def get_num_actions(self):
        return self._num_actions

    def reset(self, tasks_to_reset=None, reset_all=False):
        """ Reset behaviors:
        - If first time reset, will reset all tasks.
        - If resest_all is True, will reset all tasks.
        - Else, if tasks_to_reset is None, will only reset tasks that have kill = True
        - Else, if tasks_to_reset is given (a list of task indices), then only those tasks
            will be reset, regardless of whether or not their kill is True
        """
        if self._first_reset == True:
            self._gym.simulate(self._sim)
            self._gym.fetch_results(self._sim, True)
            self._first_reset = False
            reset_all = True

        if reset_all:
            tasks_to_reset = np.arange(self._num_envs)
        else:
            if tasks_to_reset is None:
                tasks_to_reset = np.where(self._task.dones_buffer.squeeze())[0].flatten()

        num_to_reset = tasks_to_reset.size  # np.sum(self._task.donesBuffer.astype(int))

        if num_to_reset > 0:
            return self._task.reset(tasks_to_reset)
        return np.copy(self._task.observationBuffer)

    def step(self, actions):
        # apply the actions
        for env in self.envs:
            env.step(actions)

        # simulate
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        obs, infos = self._task.fill_observations(actions)
        dones = np.copy(self._task.fill_dones(actions))
        rews = np.copy(self._task.fill_rewards(actions))

        return np.copy(obs), rews, dones, infos
