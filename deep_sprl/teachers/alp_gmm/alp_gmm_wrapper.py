from deep_sprl.teachers.abstract_teacher import BaseWrapper


class ALPGMMWrapper(BaseWrapper):

    def __init__(self, env, alp_gmm, discount_factor, context_visible):
        BaseWrapper.__init__(self, env, alp_gmm, discount_factor, context_visible)

    def done_callback(self, step):
        self.teacher.update(self.cur_context.copy(), self.discounted_reward)
