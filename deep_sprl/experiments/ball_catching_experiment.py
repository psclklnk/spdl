import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import numpy as np
import tensorflow as tf

from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class DummyBallCatchingExperiment(AbstractExperiment):
    ERROR_MSG = "Mujoco-Py is not installed. Hence running experiments is not available for the environment"

    LOWER_CONTEXT_BOUNDS = np.array([0.125 * np.pi, 0.6, 0.75])
    UPPER_CONTEXT_BOUNDS = np.array([0.5 * np.pi, 1.1, 4.])

    INITIAL_MEAN = np.array([0.68, 0.9, 0.85])
    INITIAL_VARIANCE = np.diag([1e-3, 1e-3, 0.1])

    TARGET_MEAN = np.array([0.3375 * np.pi, 0.85, 2.375])
    TARGET_VARIANCE = np.diag([0.2 * np.pi, 0.15, 1.])

    DISCOUNT_FACTOR = 0.995
    MAX_KL = 0.05

    ZETA = {Learner.TRPO: 0.4, Learner.PPO: 0.5, Learner.SAC: 0.7}
    ALPHA_OFFSET = {Learner.TRPO: 0, Learner.PPO: 0, Learner.SAC: 0}
    OFFSET = {Learner.TRPO: 5, Learner.PPO: 5, Learner.SAC: 5}

    STEPS_PER_ITER = 5000
    LAM = 0.95

    AG_P_RAND = {Learner.TRPO: 0.2, Learner.PPO: 0.3, Learner.SAC: 0.3}
    AG_FIT_RATE = {Learner.TRPO: 200, Learner.PPO: 200, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.TRPO: 2000, Learner.PPO: 2000, Learner.SAC: 1000}

    GG_NOISE_LEVEL = {Learner.TRPO: 0.1, Learner.PPO: 0.1, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.TRPO: 200, Learner.PPO: 200, Learner.SAC: 200}
    GG_P_OLD = {Learner.TRPO: 0.3, Learner.PPO: 0.3, Learner.SAC: 0.3}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)

    def create_experiment(self):
        raise RuntimeError(self.ERROR_MSG)

    def get_env_name(self):
        return "ball_catching"

    def create_self_paced_teacher(self):
        raise RuntimeError(self.ERROR_MSG)

    def evaluate_learner(self, path):
        raise RuntimeError(self.ERROR_MSG)


class BallCatchingExperiment(DummyBallCatchingExperiment):

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        if "INIT_CONTEXT" in parameters:
            self.init_context = "True" == parameters["INIT_CONTEXT"]
            del parameters["INIT_CONTEXT"]
        else:
            self.init_context = True

        if "INIT_POLICY" in parameters:
            self.init_policy = "True" == parameters["INIT_POLICY"]
            del parameters["INIT_POLICY"]
        else:
            self.init_policy = True

        if not self.init_context:
            self.ALPHA_OFFSET = {Learner.TRPO: 50, Learner.PPO: 50, Learner.SAC: 50}

        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self.eval_env = self.create_environment(evaluate=True)

    def create_environment(self, evaluate=False):
        env = gym.make("ContextualBallCatching-v1")
        if self.curriculum.default() or evaluate:
            teacher = GaussianSampler(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(),
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.goal_gan():
            if self.init_context:
                samples = np.random.multivariate_normal(self.INITIAL_MEAN, self.INITIAL_VARIANCE, size=1000)
            else:
                samples = None
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            if self.init_context:
                teacher = SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(),
                                           self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy(), bounds, alpha_fn,
                                           max_kl=self.MAX_KL, use_avg_performance=True)
            else:
                teacher = SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(),
                                           self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), bounds, alpha_fn,
                                           max_kl=self.MAX_KL, use_avg_performance=True)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                                   max_context_buffer_size=100, reset_contexts=False)
        else:
            raise RuntimeError("Invalid learning type")

        return env

    def get_other_appendix(self):
        return ("" if self.init_context else "_no_init_con") + ("" if self.init_policy else "_no_init_pol")

    def create_alg_params(self):
        params = dict(common=dict(gamma=self.DISCOUNT_FACTOR, n_cpu_tf_sess=1, seed=self.seed, verbose=0,
                                  policy_kwargs=dict(layers=[64, 64, 64], act_fun=tf.tanh)),
                      trpo=dict(max_kl=0.01, timesteps_per_batch=self.STEPS_PER_ITER, lam=self.LAM,
                                vf_stepsize=3e-4),
                      ppo=dict(n_steps=self.STEPS_PER_ITER, noptepochs=10, nminibatches=25, lam=self.LAM,
                               max_grad_norm=None, vf_coef=1.0, cliprange_vf=-1, ent_coef=0.),
                      sac=dict(learning_rate=3e-4, buffer_size=100000, learning_starts=1000, batch_size=512,
                               train_freq=1, target_entropy="auto"))

        if self.init_policy:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            params["pretrain_data_path"] = os.path.join(dir_path, "data", "expert_data.npz")

        return params

    def create_experiment(self):
        env = self.create_environment()
        model, interface = self.learner.create_learner(env, self.create_alg_params())

        timesteps = 500 * self.STEPS_PER_ITER
        if isinstance(env.teacher, SelfPacedTeacher):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {"learner": interface, "env_wrapper": env, "sp_teacher": sp_teacher, "n_inner_steps": 1,
                           "n_offset": self.OFFSET[self.learner], "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER if self.learner.sac() else 1}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        return SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                use_avg_performance=True)

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model")
        model = self.learner.load_for_evaluation(model_load_path, self.eval_env)

        for i in range(0, 200):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.eval_env.step(action)

        return self.eval_env.get_statistics()[1]
