from deep_sprl.teachers.abstract_teacher import BaseWrapper


class GoalGANWrapper(BaseWrapper):

    def __init__(self, env, goal_gan, discount_factor, context_visible):
        BaseWrapper.__init__(self, env, goal_gan, discount_factor, context_visible)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        self.teacher.update(cur_context, float(step[3]["success"]))
