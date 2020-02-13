import os
import time
import pickle
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class CurriculumType(Enum):
    GoalGAN = 1
    ALPGMM = 2
    SelfPaced = 3
    Default = 4
    Random = 5

    def __str__(self):
        if self.value == CurriculumType.GoalGAN.value:
            return "goal_gan"
        elif self.value == CurriculumType.ALPGMM.value:
            return "alp_gmm"
        elif self.value == CurriculumType.SelfPaced.value:
            return "self_paced"
        elif self.value == CurriculumType.Default.value:
            return "default"
        else:
            return "random"

    def self_paced(self):
        return self.value == CurriculumType.SelfPaced.value

    def goal_gan(self):
        return self.value == CurriculumType.GoalGAN.value

    def alp_gmm(self):
        return self.value == CurriculumType.ALPGMM.value

    def default(self):
        return self.value == CurriculumType.Default.value

    def random(self):
        return self.value == CurriculumType.Random.value

    @staticmethod
    def from_string(string):
        if string == str(CurriculumType.GoalGAN):
            return CurriculumType.GoalGAN
        elif string == str(CurriculumType.ALPGMM):
            return CurriculumType.ALPGMM
        elif string == str(CurriculumType.SelfPaced):
            return CurriculumType.SelfPaced
        elif string == str(CurriculumType.Default):
            return CurriculumType.Default
        elif string == str(CurriculumType.Random):
            return CurriculumType.Random
        else:
            raise RuntimeError("Invalid string: '" + string + "'")


class AgentInterface(ABC):

    def __init__(self, learner, obs_dim):
        self.learner = learner
        self.obs_dim = obs_dim

    @abstractmethod
    def estimate_value(self, inputs):
        pass

    @abstractmethod
    def mean_policy_std(self, cb_args, cb_kwargs):
        pass

    def save(self, log_path):
        self.learner.save(log_path)


class SACInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value(self, inputs):
        return np.squeeze(self.learner.sess.run([self.learner.step_ops[6]], {self.learner.observations_ph: inputs}))

    def mean_policy_std(self, cb_args, cb_kwargs):
        if "infos_values" in cb_args[0] and len(cb_args[0]["infos_values"]) > 0:
            return cb_args[0]["infos_values"][4]
        else:
            return np.nan


class TRPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value(self, inputs):
        return self.learner.policy_pi.value(inputs)

    def mean_policy_std(self, cb_args, cb_kwargs):
        log_std = np.squeeze(self.learner.sess.run([self.learner.policy_pi.proba_distribution.logstd],
                                                   {self.learner.policy_pi.obs_ph: np.zeros((1, self.obs_dim))})[0])
        return np.mean(np.exp(log_std))


class PPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value(self, inputs):
        return self.learner.value(inputs)

    def mean_policy_std(self, cb_args, cb_kwargs):
        log_std = np.squeeze(self.learner.sess.run([self.learner.train_model.proba_distribution.logstd],
                                                   {self.learner.train_model.obs_ph: np.zeros((1, self.obs_dim))})[0])
        return np.mean(np.exp(log_std))


class ExperimentCallback:

    def __init__(self, log_directory, learner, env_wrapper, sp_teacher=None, n_inner_steps=1, n_offset=0,
                 save_interval=5, step_divider=1):
        self.log_dir = os.path.realpath(log_directory)
        self.learner = learner
        self.env_wrapper = env_wrapper
        self.sp_teacher = sp_teacher
        self.n_offset = n_offset
        self.n_inner_steps = n_inner_steps
        self.save_interval = save_interval
        self.algorithm_iteration = 0
        self.step_divider = step_divider
        self.iteration = 0
        self.last_time = None

        self.format = "   %4d    | %.1E |   %3d    |  %.2E  |  %.2E  |  %.2E   "
        if self.sp_teacher is not None:
            context_dim = self.sp_teacher.context_dist.mean().shape[0]
            text = "| [%.2E"
            for i in range(0, context_dim - 1):
                text += ", %.2E"
            text += "] "
            self.format += text + text

        header = " Iteration |  Time   | Ep. Len. | Mean Reward | Mean Disc. Reward | Mean Policy STD "
        if self.sp_teacher is not None:
            header += "|     Context mean     |      Context std     "
        print(header)

    def estimate_value(self, inputs):
        return self.learner.estimate_value(inputs)

    def __call__(self, *args, **kwargs):
        if self.algorithm_iteration % self.step_divider == 0:
            data_tpl = (self.iteration,)

            t_new = time.time()
            dt = np.nan
            if self.last_time is not None:
                dt = t_new - self.last_time
            data_tpl += (dt,)

            mean_rew, mean_disc_rew, mean_length = self.env_wrapper.get_statistics()
            data_tpl += (int(mean_length), mean_rew, mean_disc_rew)

            data_tpl += (self.learner.mean_policy_std(args, kwargs),)

            if self.sp_teacher is not None:
                if self.iteration >= self.n_offset and self.iteration % self.n_inner_steps == 0:
                    vf_inputs, contexts = self.env_wrapper.get_context_buffer()
                    self.sp_teacher.update_distribution(mean_disc_rew, contexts, self.estimate_value(vf_inputs))
                context_mean = self.sp_teacher.context_dist.mean()
                context_std = np.sqrt(np.diag(self.sp_teacher.context_dist.covariance_matrix()))
                data_tpl += tuple(context_mean.tolist())
                data_tpl += tuple(context_std.tolist())

            print(self.format % data_tpl)

            if self.iteration % self.save_interval == 0:
                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(self.iteration))
                os.makedirs(iter_log_dir, exist_ok=True)

                self.learner.save(os.path.join(iter_log_dir, "model"))
                if self.sp_teacher is not None:
                    self.sp_teacher.save(os.path.join(iter_log_dir, "context_dist"))

            self.last_time = time.time()
            self.iteration += 1

        self.algorithm_iteration += 1


class AbstractExperiment(ABC):

    def __init__(self, base_log_dir, curriculum_name, parameters, seed, view=False):
        self.base_log_dir = base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(curriculum_name)
        self.seed = seed
        self.view = view

    @abstractmethod
    def create_experiment(self):
        pass

    @abstractmethod
    def get_env_name(self):
        pass

    @abstractmethod
    def create_self_paced_teacher(self):
        pass

    @abstractmethod
    def evaluate_learner(self, path):
        pass

    def get_log_dir(self):
        return os.path.join(self.base_log_dir, self.get_env_name(), str(self.curriculum), "seed-" + str(self.seed))

    def train(self):
        model, timesteps, callback_params = self.create_experiment()
        log_directory = self.get_log_dir()
        if os.path.exists(log_directory):
            print("Log directory already exists! Going directly to evaluation")
        else:
            callback = ExperimentCallback(log_directory=log_directory, **callback_params)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)

    def evaluate(self):
        log_dir = self.get_log_dir()

        iteration_dirs = [d for d in os.listdir(log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        sorted_iteration_dirs = np.array(iteration_dirs)[idxs].tolist()

        # First evaluate the KL-Divergences if Self-Paced learning was used
        if self.curriculum.self_paced() and not os.path.exists(os.path.join(log_dir, "kl_divergences.pkl")):
            kl_divergences = []
            for iteration_dir in sorted_iteration_dirs:
                teacher = self.create_self_paced_teacher()
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                teacher.load(os.path.join(iteration_log_dir, "context_dist.npy"))
                kl_divergences.append(teacher.target_context_kl())

            kl_divergences = np.array(kl_divergences)
            with open(os.path.join(log_dir, "kl_divergences.pkl"), "wb") as f:
                pickle.dump(kl_divergences, f)

        if not os.path.exists(os.path.join(log_dir, "performance.pkl")):
            seed_performance = []
            for iteration_dir in sorted_iteration_dirs:
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                perf = self.evaluate_learner(iteration_log_dir)
                print("Evaluated " + iteration_dir + ": " + str(perf))
                seed_performance.append(perf)

            seed_performance = np.array(seed_performance)
            with open(os.path.join(log_dir, "performance.pkl"), "wb") as f:
                pickle.dump(seed_performance, f)
