import numpy as np


class GaussianSampler:

    def __init__(self, mean, variance, bounds):
        self.bounds = bounds
        self._mean = mean
        self.covariance = variance if isinstance(variance, np.ndarray) else np.eye(self._mean.shape[0]) * variance

    def sample(self):
        return np.clip(np.random.multivariate_normal(self._mean, self.covariance), self.bounds[0], self.bounds[1])

    def mean(self):
        return self._mean.copy()

    def covariance_matrix(self):
        return self.covariance.copy()


class UniformSampler:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self):
        norm_sample = np.random.uniform(low=-1, high=1, size=self.lower_bound.shape)
        return self._scale_context(norm_sample)

    def mean(self):
        return 0.5 * self.lower_bound + 0.5 * self.upper_bound

    def covariance_matrix(self):
        return np.diag((0.5 * (self.upper_bound - self.lower_bound)) ** 2)

    def _scale_context(self, context):
        b = 0.5 * (self.upper_bound + self.lower_bound)
        m = 0.5 * (self.upper_bound - self.lower_bound)
        return m * context + b
