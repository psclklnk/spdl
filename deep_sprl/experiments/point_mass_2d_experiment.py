import os
import gym
import numpy as np

from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy

from deep_sprl.experiments.abstract_experiment import AbstractExperiment, TRPOInterface
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler, UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines.common.vec_env import DummyVecEnv


class PointMass2DExperiment(AbstractExperiment):
    LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5])
    UPPER_CONTEXT_BOUNDS = np.array([4., 8.])

    INITIAL_MEAN = np.array([0., 4.25])
    INITIAL_VARIANCE = np.diag(np.square([2, 1.875]))

    TARGET_MEAN = np.array([2.5, 0.5])
    TARGET_VARIANCE = np.diag(np.square([4e-3, 3.75e-3]))

    DISCOUNT_FACTOR = 0.95
    STD_LOWER_BOUND = np.array([0.2, 0.1875])
    KL_THRESHOLD = 8000.
    MAX_KL = 0.05

    ZETA = 1.6
    ALPHA_OFFSET = 70
    OFFSET = 5

    def __init__(self, base_log_dir, curriculum_name, parameters, seed):
        super().__init__(base_log_dir, curriculum_name, parameters, seed)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("ContextualPointMass2D-v1")
        if evaluation or self.curriculum.default():
            teacher = GaussianSampler(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE,
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=100)
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET, self.ZETA)
            bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            teacher = SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                       self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                       std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                       use_avg_performance=True)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_experiment(self):
        steps_per_iter = 2048
        timesteps = 1000 * steps_per_iter

        env, vec_env = self.create_environment(evaluation=False)
        model = TRPO(MlpPolicy, vec_env, gamma=self.DISCOUNT_FACTOR, max_kl=0.004, timesteps_per_batch=steps_per_iter,
                     lam=0.99, policy_kwargs=dict(net_arch=[dict(vf=[21], pi=[21])]), vf_stepsize=0.23921693516009684,
                     seed=self.seed, n_cpu_tf_sess=1)

        if isinstance(env.teacher, SelfPacedTeacher):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {"learner": TRPOInterface(model, env.observation_space.shape[0]), "env_wrapper": env,
                           "sp_teacher": sp_teacher, "n_inner_steps": 1, "n_offset": self.OFFSET, "save_interval": 5,
                           "step_divider": 1}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET, self.ZETA)
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        return SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                std_lower_bound=self.STD_LOWER_BOUND, kl_threshold=self.KL_THRESHOLD,
                                use_avg_performance=True)

    def get_env_name(self):
        return "point_mass_2d"

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model.zip")
        model = TRPO.load(model_load_path, env=self.vec_eval_env)
        for i in range(0, 50):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, None, done)[0]
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_statistics()[1]
