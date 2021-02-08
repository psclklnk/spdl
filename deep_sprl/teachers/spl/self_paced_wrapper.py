import numpy as np
from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class SelfPacedWrapper(BaseWrapper):

    def __init__(self, env, sp_teacher, discount_factor, context_visible, max_context_buffer_size=1000,
                 reset_contexts=True):
        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible)

        self.context_buffer = Buffer(3, max_context_buffer_size, reset_contexts)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        self.context_buffer.update_buffer((cur_initial_state, cur_context, discounted_reward))

    def get_context_buffer(self):
        ins, cons, rewards = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(rewards)