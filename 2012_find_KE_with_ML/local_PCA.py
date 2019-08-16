import numpy as np


class Local_PCA(object):
    
    @staticmethod
    def _nearest(x, x1, nk):
        sq_dist = np.sqrt(np.sum((x - x1)**2, axis=-1))
        topk_index = np.argpartition(sq_dist, nk)[:nk]
        return topk_index
        
    def __init__(self, n_components, k_nearest, train_x):
        self.n_components = n_components
        self.k_nearest = k_nearest
        self.train_x = train_x

    def __call__(self, x):
        x_ = np.atleast_2d(x)
        topk_index = self._nearest(x_, self.train_x, self.k_nearest)
        topk_train_x = self.train_x[topk_index]
        X = topk_train_x - x_
        U, _, _ = np.linalg.svd(X.T / np.sqrt(self.k_nearest))
        U_proj = U[:, :self.n_components]
        return U_proj

