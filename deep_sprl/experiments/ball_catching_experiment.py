import os
import gym
import numpy as np
import tensorflow as tf

from stable_baselines.sac import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.gail.dataset.dataset import ExpertDataset

from deep_sprl.experiments.abstract_experiment import AbstractExperiment, SACInterface
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class BallCatchingExperiment(AbstractExperiment):
    LOWER_CONTEXT_BOUNDS = np.array([0.125 * np.pi, 0.6, 0.75])
    UPPER_CONTEXT_BOUNDS = np.array([0.5 * np.pi, 1.1, 4.])

    INITIAL_MEAN = np.array([0.68, 0.9, 0.85])
    INITIAL_VARIANCE = np.diag([1e-3, 1e-3, 0.1])

    TARGET_MEAN = np.array([0.3375 * np.pi, 0.85, 2.375])
    TARGET_VARIANCE = np.diag([0.2 * np.pi, 0.15, 1.])

    DISCOUNT_FACTOR = 0.995
    MAX_KL = 0.05

    ZETA = 0.7
    ALPHA_OFFSET = 0
    OFFSET = 35

    def __init__(self, base_log_dir, curriculum_name, parameters, seed):
        super().__init__(base_log_dir, curriculum_name, parameters, seed)
        self.eval_env = self.create_environment(evaluate=True)

    def create_environment(self, evaluate=False):
        env = gym.make("ContextualBallCatching-v1")
        if self.curriculum.default() or evaluate:
            teacher = GaussianSampler(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(),
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.goal_gan():
            samples = np.random.multivariate_normal(self.INITIAL_MEAN, self.INITIAL_VARIANCE, size=1000)
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=0.05, success_distance_threshold=0.01, update_size=125, n_rollouts=2,
                              goid_lb=0.25, goid_ub=0.75, p_old=0.2, pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET, self.ZETA)
            bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            teacher = SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                       self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                       use_avg_performance=True)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                                   max_context_buffer_size=100, reset_contexts=False)
        else:
            raise RuntimeError("Invalid learning type")

        return env

    def create_experiment(self):
        env = self.create_environment()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(dir_path, "data", "expert_data.npz")
        model = SAC(MlpPolicy, env, verbose=0, gamma=self.DISCOUNT_FACTOR, learning_rate=3e-4, buffer_size=100000,
                    learning_starts=1000, batch_size=512, train_freq=1, target_entropy="auto",
                    policy_kwargs={"layers": [64, 64, 64], "act_fun": tf.tanh}, seed=self.seed,
                    n_cpu_tf_sess=1)
        model.pretrain(ExpertDataset(expert_path=data_path, verbose=0), n_epochs=25)

        timesteps = 2500000
        if isinstance(env.teacher, SelfPacedTeacher):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {"learner": SACInterface(model, env.observation_space.shape[0]), "env_wrapper": env,
                           "sp_teacher": sp_teacher, "n_inner_steps": 1, "n_offset": self.OFFSET, "save_interval": 5,
                           "step_divider": 5000}
        return model, timesteps, callback_params

    def get_env_name(self):
        return "ball_catching"

    def create_self_paced_teacher(self):
        alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET, self.ZETA)
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        return SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE.copy(), self.INITIAL_MEAN.copy(),
                                self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.MAX_KL,
                                use_avg_performance=True)

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model")
        model = SAC.load(model_load_path, self.eval_env)

        for i in range(0, 200):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action = model.predict(obs, deterministic=False)[0]
                obs, rewards, done, infos = self.eval_env.step(action)

        return self.eval_env.get_statistics()[1]
