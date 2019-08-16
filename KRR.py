import autograd.numpy as np 
from scipy.linalg import cholesky, cho_solve

__all__ = ['RBFKenrel', 'KernelRidge']

class RBFKernel(object):
    @staticmethod
    def _sq_dist(x, x1):
        n_points = x.shape[1]
        delta_x = 1.0 / n_points

        sq_dist = np.sum((x[:, np.newaxis, :] - x1)**2, axis=-1) * delta_x
        return sq_dist
    
    def __init__(self, gamma, train_x):
        self.gamma = gamma
        self.train_x = train_x

    def __call__(self, x):
        sq_dist = self._sq_dist(x, self.train_x)
        return np.exp(-1 * self.gamma * sq_dist)



class KernelRidge(object):
    def __init__(self, gamma, sigma, train_x, train_y):
        self.gamma = gamma
        self.sigma = sigma
        self.train_x = train_x
        self.train_y = train_y
        self.kernel_fn = RBFKernel(gamma, train_x)

    def fit(self):
        n_samples = len(self.train_x)

        K_XX = self.kernel_fn(self.train_x) + (self.sigma**2) * np.eye(n_samples)
        L = cholesky(K_XX, lower=True)
        alpha = cho_solve((L, True), self.train_y)

        self.alpha = alpha
        return None

    def __call__(self, x):
        K_xX = self.kernel_fn(np.atleast_2d(x))
        return K_xX @ self.alpha


# class KRR_gradient(object):
#     def __init__(self, KRR_model):
#         self.gamma = KRR_model.gamma
#         self.alpha = KRR_model.alpha
#         self.train_x = KRR_model.train_x
#         self.kernel_fn = KRR_model.kernel_fn

#     def __call__(self, x):
#         x_2d = np.atleast_2d(x)
#         n_points = x_2d.shape[1]
#         diff = x_2d - self.train_x
#         K_xX = self.kernel_fn(x_2d)
#         return (-2*self.gamma) * ((diff.T * K_xX ) @ self.alpha)