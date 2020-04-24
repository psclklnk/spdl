import numpy as np
import time

from gym import Env, spaces
from deep_sprl.util.viewer import Viewer


class ContextualPointMass(Env):

    def __init__(self, context=np.array([0., 2., 2.])):
        self.action_space = spaces.Box(np.array([-10., -10.]), np.array([10., 10.]))
        self.observation_space = spaces.Box(np.array([-4., -np.inf, -4., -np.inf]),
                                            np.array([4., np.inf, 4., np.inf]))

        self._state = None
        self._goal_state = np.array([0., 0., -3., 0.])
        self.context = context
        self._dt = 0.01
        self._viewer = Viewer(8, 8, background=(255, 255, 255))

    def reset(self):
        self._state = np.array([0., 0., 3., 0.])
        return np.copy(self._state)

    def _step_internal(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_der = np.zeros(4)
        state_der[0::2] = state[1::2]
        friction_param = self.context[2]
        state_der[1::2] = 1.5 * action - friction_param * state[1::2] + np.random.normal(0, 0.05, (2,))
        new_state = np.clip(state + self._dt * state_der, self.observation_space.low,
                            self.observation_space.high)

        crash = False
        if state[2] >= 0 > new_state[2] or state[2] <= 0 < new_state[2]:
            alpha = (0. - state[2]) / (new_state[2] - state[2])
            x_crit = alpha * new_state[0] + (1 - alpha) * state[0]

            if np.abs(x_crit - self.context[0]) > 0.5 * self.context[1]:
                new_state = np.array([x_crit, 0., 0., 0.])
                crash = True

        return new_state, crash

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        new_state = self._state
        crash = False
        for i in range(0, 10):
            new_state, crash = self._step_internal(new_state, action)
            if crash:
                break

        self._state = np.copy(new_state)

        info = {"success": np.linalg.norm(self._goal_state[0::2] - new_state[0::2]) < 0.25}

        return new_state, np.exp(-0.6 * np.linalg.norm(self._goal_state[0::2] - new_state[0::2])), crash, info

    def render(self, mode='human'):
        pos = self.context[0] + 4.
        width = self.context[1]
        self._viewer.line(np.array([0., 4.]), np.array([np.clip(pos - 0.5 * width, 0., 8.), 4.]), color=(0, 0, 0),
                          width=0.2)
        self._viewer.line(np.array([np.clip(pos + 0.5 * width, 0., 8, ), 4.]), np.array([8., 4.]), color=(0, 0, 0),
                          width=0.2)

        self._viewer.line(np.array([3.9, 0.9]), np.array([4.1, 1.1]), color=(255, 0, 0), width=0.1)
        self._viewer.line(np.array([4.1, 0.9]), np.array([3.9, 1.1]), color=(255, 0, 0), width=0.1)

        self._viewer.circle(self._state[0::2] + np.array([4., 4.]), 0.1, color=(0, 0, 0))
        self._viewer.display(self._dt)
