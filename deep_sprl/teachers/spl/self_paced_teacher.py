import torch
import numpy as np
from copy import deepcopy
from functools import partial
from deep_sprl.util.conjugate_gradient import cg_step
from deep_sprl.util.torch import to_float_tensor
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution
from deep_sprl.teachers.abstract_teacher import AbstractTeacher


class AbstractSelfPacedTeacher:

    def __init__(self, init_mean, flat_init_chol, target_mean, flat_target_chol, alpha_function, max_kl, cg_parameters):
        self.context_dist = GaussianTorchDistribution(init_mean, flat_init_chol, use_cuda=False)
        self.target_dist = GaussianTorchDistribution(target_mean, flat_target_chol, use_cuda=False)

        self.alpha_function = alpha_function
        self.max_kl = max_kl
        self.cg_parameters = {"n_epochs_line_search": 10, "n_epochs_cg": 10, "cg_damping": 1e-2,
                              "cg_residual_tol": 1e-10}
        if cg_parameters is not None:
            self.cg_parameters.update(cg_parameters)

        self.task = None
        self.iteration = 0

    def target_context_kl(self, numpy=True):
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist.distribution_t,
                                                      self.target_dist.distribution_t).detach()
        if numpy:
            kl_div = kl_div.numpy()

        return kl_div

    def save(self, path):
        weights = self.context_dist.get_weights()
        np.save(path, weights)

    def load(self, path):
        self.context_dist.set_weights(np.load(path))

    def _compute_context_kl(self, old_context_dist):
        return torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, self.context_dist.distribution_t)

    def _compute_context_loss(self, cons_t, old_c_log_prob_t, c_val_t, alpha_cur_t):
        con_ratio_t = torch.exp(self.context_dist.log_pdf_t(cons_t) - old_c_log_prob_t)
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist.distribution_t, self.target_dist.distribution_t)
        return torch.mean(con_ratio_t * c_val_t) - alpha_cur_t * kl_div


class SelfPacedTeacher(AbstractTeacher, AbstractSelfPacedTeacher):

    def __init__(self, target_mean, target_variance, initial_mean, initial_variance, context_bounds, alpha_function,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, cg_parameters=None, use_avg_performance=False):

        # The bounds that we show to the outside are limited to the interval [-1, 1], as this is typically better for
        # neural nets to deal with
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        self.use_avg_performance = use_avg_performance

        if std_lower_bound is not None and kl_threshold is None:
            raise RuntimeError("Error! Both Lower Bound on standard deviation and kl threshold need to be set")
        else:
            if std_lower_bound is not None:
                if isinstance(std_lower_bound, np.ndarray):
                    if std_lower_bound.shape[0] != self.context_dim:
                        raise RuntimeError("Error! Wrong dimension of the standard deviation lower bound")
                elif std_lower_bound is not None:
                    std_lower_bound = np.ones(self.context_dim) * std_lower_bound
            self.std_lower_bound = std_lower_bound
            self.kl_threshold = kl_threshold

        # Create the initial context distribution
        if isinstance(initial_variance, np.ndarray):
            flat_init_chol = GaussianTorchDistribution.flatten_matrix(initial_variance, tril=False)
        else:
            flat_init_chol = GaussianTorchDistribution.flatten_matrix(initial_variance * np.eye(self.context_dim),
                                                                      tril=False)

        # Create the target distribution
        if isinstance(target_variance, np.ndarray):
            flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance, tril=False)
        else:
            flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance * np.eye(self.context_dim),
                                                                        tril=False)

        super(SelfPacedTeacher, self).__init__(initial_mean, flat_init_chol, target_mean, flat_target_chol,
                                               alpha_function, max_kl, cg_parameters)

    def update_distribution(self, avg_performance, contexts, values):
        self.iteration += 1

        old_context_dist = deepcopy(self.context_dist)
        contexts_t = to_float_tensor(contexts, use_cuda=False)
        old_c_log_prob_t = old_context_dist.log_pdf_t(contexts_t).detach()

        # Estimate the value of the state after the policy update
        c_val_t = to_float_tensor(values, use_cuda=False)

        # Add the penalty term
        cur_kl_t = self.target_context_kl(numpy=False)
        if self.use_avg_performance:
            alpha_cur_t = self.alpha_function(self.iteration, avg_performance, cur_kl_t)
        else:
            alpha_cur_t = self.alpha_function(self.iteration, torch.mean(c_val_t).detach(), cur_kl_t)

        cg_step(partial(self._compute_context_loss, contexts_t, old_c_log_prob_t, c_val_t, alpha_cur_t),
                partial(self._compute_context_kl, old_context_dist), self.max_kl,
                self.context_dist.parameters, self.context_dist.set_weights,
                self.context_dist.get_weights, **self.cg_parameters, use_cuda=False)

        if self.std_lower_bound is not None and self.target_context_kl() > self.kl_threshold:
            cov = self.context_dist._chol_flat.detach().numpy()
            cov[0:self.context_dim] = np.log(np.maximum(np.exp(cov[0:self.context_dim]), self.std_lower_bound))
            self.context_dist.set_weights(np.concatenate((self.context_dist.mean(), cov)))

    def sample(self):
        sample = self.context_dist.sample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])
