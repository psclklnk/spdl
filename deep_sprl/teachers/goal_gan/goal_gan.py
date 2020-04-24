import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from queue import Queue
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from deep_sprl.teachers.goal_gan.generator import StateGAN, StateCollection


class GoalGAN(AbstractTeacher):

    def __init__(self, mins, maxs, state_noise_level, success_distance_threshold, update_size, n_rollouts=2,
                 goid_lb=0.25, goid_ub=0.75, p_old=0.2, pretrain_samples=None):
        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        # Prevent tensorflow from taking all the gpu memory
        tf_config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(config=tf_config)
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

        self.pending_contexts = {}
        self.context_queue = Queue()

        if pretrain_samples is not None:
            self.gan.pretrain(pretrain_samples)

    def sample(self):
        if self.context_queue.empty():
            if np.random.random() > self.p_old or self.success_buffer.size == 0:
                context = self.gan.sample_states_with_noise(1)[0][0, :]
                context_key = context.tobytes()

                # Either extend or create the list - note that this is a nested list to support the case when an
                # identical context is sampled twice (Happens e.g. due to clipping)
                if context_key in self.pending_contexts:
                    self.pending_contexts[context_key].append([])
                else:
                    self.pending_contexts[context_key] = [[]]

                # Store the contexts in the buffer for being sampled
                for i in range(0, self.n_rollouts - 1):
                    self.context_queue.put(context.copy())
            else:
                context = self.success_buffer.sample(size=1, replay_noise=self.replay_noise)[0, :]
        else:
            context = self.context_queue.get()

        return context

    def update(self, context, success):
        context_key = context.tobytes()
        if context_key in self.pending_contexts:
            # Always append to the first list
            self.pending_contexts[context_key][0].append(success)

            if len(self.pending_contexts[context_key][0]) >= self.n_rollouts:
                mean_success = np.mean(self.pending_contexts[context_key][0])
                self.labels.append(self.goid_lb <= mean_success <= self.goid_ub)
                self.contexts.append(context.copy())

                if mean_success > self.goid_ub:
                    self.success_buffer.append(context.copy()[None, :])

                # Delete the first entry of the nested list
                del self.pending_contexts[context_key][0]

                # If the list is now empty, we can remove the whole entry in the map
                if len(self.pending_contexts[context_key]) == 0:
                    del self.pending_contexts[context_key]

        if len(self.contexts) >= self.update_size:
            labels = np.array(self.labels, dtype=np.float)[:, None]
            if np.any(labels):
                print("Training GoalGAN with " + str(len(self.contexts)) + " contexts")
                self.gan.train(np.array(self.contexts), labels, 250)
            else:
                print("No positive samples in " + str(len(self.contexts)) + " contexts - skipping GoalGAN training")

            self.contexts = []
            self.labels = []
