import numpy as np

from abc import ABC, abstractmethod
from anisotropic import function as prox
import compot.calculus.function as fun

class ProxFunction(ABC):
    @abstractmethod
    def eval(self, x):
        pass

    @abstractmethod
    def eval_conjugate(self, x):
        pass

    @abstractmethod
    def eval_grad(self, x):
        pass

    @abstractmethod
    def eval_grad_conjugate(self, x):
        pass

    @abstractmethod
    def eval_prox(self, y, v):
        pass

class NormpPowerProxFunction(ProxFunction):
    def __init__(self, p, lamb = 1.):
        self.p = p
        self.q = p / (p - 1.)
        self.lamb = lamb

    def eval(self, x):
        return self.lamb * np.sum(np.power(np.abs(x / self.lamb), self.p)) / self.p

    def eval_conjugate(self, x):
        return self.lamb * np.sum(np.power(np.abs(x), self.q)) / self.q

    def eval_grad(self, x):
        return np.sign(x) * np.power(np.abs(x / self.lamb), self.p - 1)

    def eval_grad_conjugate(self, x):
        return self.lamb * np.sign(x) * np.power(np.abs(x), self.q - 1)

class NormpPowerProxFunctionBox(NormpPowerProxFunction):
    def eval_prox(self, y, v):
        return np.minimum(1., np.maximum(-1., y + self.eval_grad_conjugate(v)))

class Norm2PowerProxFunction(ProxFunction):
    def __init__(self, p, proxable = None, lamb = 1.):
        self.p = p
        self.q = p / (p - 1.)
        self.tau = 1.
        self.proxable = proxable
        self.lamb = lamb

    def eval(self, x):
        return self.lamb * np.power(np.linalg.norm(x / self.lamb, 2), self.p) / self.p

    def eval_conjugate(self, x):
        return self.lamb * np.power(np.linalg.norm(x, 2), self.q) / self.q

    def eval_grad(self, x):
        return np.power(np.linalg.norm(x / self.lamb, 2), self.p - 2) * x / self.lamb

    def eval_grad_conjugate(self, x):
        return self.lamb * np.power(np.linalg.norm(x, 2), self.q - 2) * x

    def eval_prox(self, y, v):
        residual = lambda tau: tau - np.power(np.linalg.norm(self.proxable.eval_prox(y + 1. / tau * v, 1. / tau) - y),
                                              self.p - 2) * np.power(self.lamb, 1. - self.p)

        #residual = lambda tau: tau - self.proxfun.elem_proxfun.eval_grad(
        #    np.linalg.norm(self.proxable.eval_prox(y + 1. / tau * v, 1. / tau) - y) / scaling
        #)

        tau_high = self.tau
        while True:
            if residual(tau_high) > 0:
                break

            tau_high = tau_high * 2

        tau_low = self.tau
        while True:
            if residual(tau_low) < 0:
                break

            tau_low = tau_low / 2

        for k in range(100):
            self.tau = tau_low + (tau_high - tau_low) / 2
            res = residual(self.tau)
            #if residual(self.tau) < 1e-15:
            #    break

            if res < 0:
                tau_low = self.tau
            elif res > 0:
                tau_high = self.tau
            else:
                break

        return self.proxable.eval_prox(y + 1. / self.tau * v, 1 / self.tau)



lamb = 1
p = 7.
proxfun_dual = prox.IsotropicProxFunction(
    prox.PowerElemProxFunction(p)
)
proxable_dual = prox.IsotropicProxable(fun.IndicatorBox(), proxfun_dual)

proxfun_dual2 = Norm2PowerProxFunction(
    p,
    fun.IndicatorBox(),
    lamb=lamb
)
np.random.seed(12)
y = 50*np.random.rand(250)-25
v = 50*np.random.rand(250)-25

x1 = proxfun_dual2.eval_prox(y, v)
x2 = proxable_dual.eval_prox(y, v, lamb)

print(x1 - x2)

print(np.linalg.norm(proxfun_dual.eval_grad(x1, scaling=lamb) - proxfun_dual2.eval_grad(x1)))

print(np.linalg.norm(proxfun_dual.eval_grad_conjugate(x1, scaling=lamb) - proxfun_dual2.eval_grad_conjugate(x1)))
print(np.linalg.norm(proxfun_dual.eval(x1, scaling=lamb) - proxfun_dual2.eval(x1)))
print(np.linalg.norm(proxfun_dual.eval_conjugate(x1, scaling=lamb) - proxfun_dual2.eval_conjugate(x1)))


