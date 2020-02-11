import numpy as np
from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class SelfPacedWrapper(BaseWrapper):

    def __init__(self, env, sp_teacher, discount_factor, context_visible, max_context_buffer_size=1000,
                 reset_contexts=True):
        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible)

        self.context_buffer = Buffer(2, max_context_buffer_size, reset_contexts)

    def done_callback(self, step):
        self.context_buffer.update_buffer((self.cur_initial_state, self.cur_context))

    def get_context_buffer(self):
        ins, cons = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons)
