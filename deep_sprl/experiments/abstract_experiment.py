import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import pickle
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from deep_sprl.util.vec_normalize import VecNormalize
from deep_sprl.util.parameter_parser import create_override_appendix

from stable_baselines.sac import SAC
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy


class CurriculumType(Enum):
    GoalGAN = 1
    ALPGMM = 2
    SelfPaced = 3
    Default = 4
    Random = 5

    def __str__(self):
        if self.goal_gan():
            return "goal_gan"
        elif self.alp_gmm():
            return "alp_gmm"
        elif self.self_paced():
            return "self_paced"
        elif self.default():
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

    def estimate_value(self, inputs):
        if isinstance(self.learner.env, VecNormalize):
            return self.estimate_value_internal(self.learner.env.normalize_obs(inputs))
        else:
            return self.estimate_value_internal(inputs)

    @abstractmethod
    def estimate_value_internal(self, inputs):
        pass

    @abstractmethod
    def mean_policy_std(self, cb_args, cb_kwargs):
        pass

    def save(self, log_dir):
        self.learner.save(os.path.join(log_dir, "model"))
        if isinstance(self.learner.env, VecNormalize):
            self.learner.env.save(os.path.join(log_dir, "normalizer.pkl"))


class SACInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value_internal(self, inputs):
        return np.squeeze(self.learner.sess.run([self.learner.step_ops[6]], {self.learner.observations_ph: inputs}))

    def mean_policy_std(self, cb_args, cb_kwargs):
        if "infos_values" in cb_args[0] and len(cb_args[0]["infos_values"]) > 0:
            return cb_args[0]["infos_values"][4]
        else:
            return np.nan


class TRPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value_internal(self, inputs):
        return self.learner.policy_pi.value(inputs)

    def mean_policy_std(self, cb_args, cb_kwargs):
        log_std = np.squeeze(self.learner.sess.run([self.learner.policy_pi.proba_distribution.logstd],
                                                   {self.learner.policy_pi.obs_ph: np.zeros((1, self.obs_dim))})[0])
        return np.mean(np.exp(log_std))


class PPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value_internal(self, inputs):
        return self.learner.value(inputs)

    def mean_policy_std(self, cb_args, cb_kwargs):
        log_std = np.squeeze(self.learner.sess.run([self.learner.train_model.proba_distribution.logstd],
                                                   {self.learner.train_model.obs_ph: np.zeros((1, self.obs_dim))})[0])
        return np.mean(np.exp(log_std))


class SACEvalWrapper:

    def __init__(self, model):
        self.model = model

    def step(self, observation, state=None, deterministic=False):
        return self.model.predict(observation, state=state, deterministic=deterministic)[0]


class PPOTRPOEvalWrapper:

    def __init__(self, model):
        self.model = model

    def step(self, observation, state=None, deterministic=False):
        if len(observation.shape) == 1:
            observation = observation[None, :]
            return self.model.step(observation, state=state, deterministic=deterministic)[0][0, :]
        else:
            return self.model.step(observation, state=state, deterministic=deterministic)[0]


class Learner(Enum):
    TRPO = 1
    PPO = 2
    SAC = 3

    def __str__(self):
        if self.trpo():
            return "trpo"
        elif self.ppo():
            return "ppo"
        else:
            return "sac"

    def trpo(self):
        return self.value == Learner.TRPO.value

    def ppo(self):
        return self.value == Learner.PPO.value

    def sac(self):
        return self.value == Learner.SAC.value

    def create_learner(self, env, parameters):
        if (self.trpo() or self.ppo()) and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])

        if self.trpo():
            model = TRPO(MlpPolicy, env, **parameters["common"], **parameters[str(self)])
            interface = TRPOInterface(model, env.observation_space.shape[0])
        elif self.ppo():
            model = PPO2(MlpPolicy, env, **parameters["common"], **parameters[str(self)])
            interface = PPOInterface(model, env.observation_space.shape[0])
        else:
            model = SAC(SACMlpPolicy, env, **parameters["common"], **parameters[str(self)])
            interface = SACInterface(model, env.observation_space.shape[0])

        if "pretrain_data_path" in parameters:
            data_path = parameters["pretrain_data_path"]
            model.pretrain(ExpertDataset(expert_path=data_path, verbose=0), n_epochs=25)

        return model, interface

    def load(self, path, env):
        if self.trpo():
            return TRPO.load(path, env=env)
        elif self.ppo():
            return PPO2.load(path, env=env)
        else:
            return SAC.load(path, env=env)

    def load_for_evaluation(self, path, env):
        if (self.trpo() or self.ppo()) and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])
        model = self.load(path, env)
        if self.sac():
            return SACEvalWrapper(model)
        else:
            return PPOTRPOEvalWrapper(model)

    @staticmethod
    def from_string(string):
        if string == str(Learner.TRPO):
            return Learner.TRPO
        elif string == str(Learner.PPO):
            return Learner.PPO
        elif string == str(Learner.SAC):
            return Learner.SAC
        else:
            raise RuntimeError("Invalid string: '" + string + "'")


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
                    self.sp_teacher.update_distribution(mean_disc_rew, contexts, self.learner.estimate_value(vf_inputs))
                context_mean = self.sp_teacher.context_dist.mean()
                context_std = np.sqrt(np.diag(self.sp_teacher.context_dist.covariance_matrix()))
                data_tpl += tuple(context_mean.tolist())
                data_tpl += tuple(context_std.tolist())

            print(self.format % data_tpl)

            if self.iteration % self.save_interval == 0:
                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(self.iteration))
                os.makedirs(iter_log_dir, exist_ok=True)

                self.learner.save(iter_log_dir)
                if self.sp_teacher is not None:
                    self.sp_teacher.save(os.path.join(iter_log_dir, "context_dist"))

            self.last_time = time.time()
            self.iteration += 1

        self.algorithm_iteration += 1


class AbstractExperiment(ABC):
    APPENDIX_KEYS = {"default": ["DISCOUNT_FACTOR", "STEPS_PER_ITER", "LAM"],
                     CurriculumType.SelfPaced: ["ALPHA_OFFSET", "MAX_KL", "OFFSET", "ZETA"],
                     CurriculumType.GoalGAN: ["GG_NOISE_LEVEL", "GG_FIT_RATE", "GG_P_OLD"],
                     CurriculumType.ALPGMM: ["AG_P_RAND", "AG_FIT_RATE", "AG_MAX_SIZE"],
                     CurriculumType.Random: [],
                     CurriculumType.Default: []}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, view=False):
        self.base_log_dir = base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(curriculum_name)
        self.learner = Learner.from_string(learner_name)
        self.seed = seed
        self.view = view
        self.process_parameters()

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

    def get_other_appendix(self):
        return ""

    @staticmethod
    def parse_max_size(val):
        if val == "None":
            return None
        else:
            return int(val)

    @staticmethod
    def parse_n_hidden(val):
        val = val.replace(" ", "")
        if not (val.startswith("[") and val.endswith("]")):
            raise RuntimeError("Invalid list specifier: " + str(val))
        else:
            vals = val[1:-1].split(",")
            res = []
            for v in vals:
                res.append(int(v))
            return res

    def process_parameters(self):
        allowed_overrides = {"DISCOUNT_FACTOR": float, "MAX_KL": float, "ZETA": float, "ALPHA_OFFSET": int,
                             "OFFSET": int, "STEPS_PER_ITER": int, "LAM": float, "AG_P_RAND": float, "AG_FIT_RATE": int,
                             "AG_MAX_SIZE": self.parse_max_size, "GG_NOISE_LEVEL": float, "GG_FIT_RATE": int,
                             "GG_P_OLD": float}
        for key in sorted(self.parameters.keys()):
            if key not in allowed_overrides:
                raise RuntimeError("Parameter '" + str(key) + "'not allowed'")

            value = self.parameters[key]
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp[self.learner] = allowed_overrides[key](value)
            else:
                setattr(self, key, allowed_overrides[key](value))

    def get_log_dir(self):
        override_appendix = create_override_appendix(self.APPENDIX_KEYS["default"], self.parameters)
        leaner_string = str(self.learner)
        key_list = self.APPENDIX_KEYS[self.curriculum]
        for key in sorted(key_list):
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp = tmp[self.learner]
            leaner_string += "_" + key + "=" + str(tmp).replace(" ", "")

        return os.path.join(self.base_log_dir, self.get_env_name(), str(self.curriculum),
                            leaner_string + override_appendix + self.get_other_appendix(), "seed-" + str(self.seed))

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
