import numpy as np
import tensorflow as tf
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from deep_sprl.teachers.goal_gan.generator import StateGAN, StateCollection


class GoalGAN(AbstractTeacher):

    def __init__(self, mins, maxs, state_noise_level, success_distance_threshold, update_size, n_rollouts=2,
                 goid_lb=0.25, goid_ub=0.75, p_old=0.2, pretrain_samples=None):
        self.tf_session = tf.Session()
        self.gan = StateGAN(
            state_size=len(mins),
            evaluater_size=1,
            state_range=0.5 * (maxs - mins),
            state_center=mins + 0.5 * (maxs - mins),
            state_noise_level=(state_noise_level * (maxs - mins))[None, :],
            generator_layers=[256, 256],
            discriminator_layers=[128, 128],
            noise_size=mins.shape[0],
            tf_session=self.tf_session,
            configs={"supress_all_logging": True}
        )
        self.tf_session.run(tf.initialize_local_variables())
        self.replay_noise = state_noise_level * (maxs - mins)
        self.success_buffer = StateCollection(1, success_distance_threshold * np.linalg.norm(maxs - mins))

        self.update_size = update_size
        self.contexts = []
        self.labels = []

        self.p_old = p_old
        self.n_rollouts = n_rollouts
        self.goid_lb = goid_lb
        self.goid_ub = goid_ub
        self.cur_context = None
        self.cur_successes = []
        self.is_replay = False

        if pretrain_samples is not None:
            self.gan.pretrain(pretrain_samples)

    def sample(self):
        if self.cur_context is None:
            if np.random.random() > self.p_old or self.success_buffer.size == 0:
                self.cur_context = self.gan.sample_states_with_noise(1)[0][0, :]
                self.is_replay = False
            else:
                self.cur_context = self.success_buffer.sample(size=1, replay_noise=self.replay_noise)[0, :]
                self.is_replay = True

        return self.cur_context.copy()

    def update(self, success):
        if self.is_replay:
            self.cur_context = None
            self.cur_successes = []
            self.is_replay = False
        else:
            self.cur_successes.append(success)
            if len(self.cur_successes) >= self.n_rollouts:
                mean_success = np.mean(self.cur_successes)
                self.labels.append(self.goid_lb <= mean_success <= self.goid_ub)
                self.cur_successes = []

                if mean_success > self.goid_ub:
                    self.success_buffer.append(self.cur_context.copy()[None, :])

                self.contexts.append(self.cur_context)
                self.cur_context = None

        if len(self.contexts) >= self.update_size:
            labels = np.array(self.labels, dtype=np.float)[:, None]
            if np.any(labels):
                print("Training GoalGAN with " + str(len(self.contexts)) + " contexts")
                self.gan.train(np.array(self.contexts), labels, 250)
            else:
                print("No positive samples in " + str(len(self.contexts)) + " contexts - skipping GoalGAN training")

            self.contexts = []
            self.labels = []
