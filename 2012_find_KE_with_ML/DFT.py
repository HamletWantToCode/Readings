import numpy as onp
from autograd import grad
from local_PCA import Local_PCA


class Energy(object):
    def __init__(self, Ek_model):
        self.Ek_model = Ek_model

    def __call__(self, x, Vx):
        dens = x
        n_points = len(dens)
        delta_x = 1.0 / n_points

        Ek = self.Ek_model(dens)
        V = onp.sum(Vx * dens) * delta_x
        return Ek[0] + V


class EnergyDer(object):
    def __init__(self, Ek_model, projector=None):
        self.dEk_model = grad(Ek_model)
        self.projector = projector

    def __call__(self, x, Vx):
        dens = x
        n_points = len(dens)
        delta_x = 1.0 / n_points

        _der = self.dEk_model(dens) * n_points + Vx
        if self.projector is not None:
            Uproj = self.projector(dens)
            der = Uproj @ Uproj.T @ _der
        else:
            der = _der
        return der


