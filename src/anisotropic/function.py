import numpy as np
import compot.calculus.function as fun

from abc import ABC, abstractmethod

##
# New
##
class ElemProxFunction(ABC):
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

class PowerElemProxFunction(ElemProxFunction):
    def __init__(self, p):
        self.p = p
        self.q = p / (p - 1.)

    def eval(self, x):
        return np.power(np.abs(x), self.p) / self.p

    def eval_conjugate(self, x):
        return np.power(np.abs(x), self.q) / self.q

    def eval_grad(self, x):
        return np.sign(x) * np.power(np.abs(x), self.p - 1.)

    def eval_grad_conjugate(self, x):
        return np.sign(x) * np.power(np.abs(x), self.q - 1.)

class CoshElemProxFunction(ElemProxFunction):
    def eval(self, x):
        return np.cosh(x)

    def eval_conjugate(self, x):
        return x * np.log(np.sqrt(x * x + 1) + x) - np.sqrt(x * x + 1)

    def eval_grad(self, x):
        return np.sinh(x)

    def eval_grad_conjugate(self, x):
        return np.log(np.sqrt(x * x + 1) + x)

class SymmetricExpElemProxFunction(ElemProxFunction):
    def __init__(self, rho):
        self.rho = rho

    def eval(self, x):
        return self.rho * (np.exp(np.abs(x)) - np.abs(x) - 1)

    def eval_conjugate(self, x):
        return (np.abs(x) + self.rho) * np.log(np.abs(x) / self.rho + 1) - np.abs(x)

    def eval_grad(self, x):
        return self.rho * np.sign(x) * (np.exp(np.abs(x)) - 1)

    def eval_grad_conjugate(self, x):
        return np.sign(x) * (np.log(np.abs(x) / self.rho + 1))

class ProxFunction(ABC):
    @abstractmethod
    def eval(self, x, scaling):
        pass

    @abstractmethod
    def eval_conjugate(self, x, scaling):
        pass

    @abstractmethod
    def eval_grad(self, x, scaling):
        pass

    @abstractmethod
    def eval_grad_conjugate(self, x, scaling):
        pass

class IsotropicProxFunction(ProxFunction):
    def __init__(self, elem_proxfun):
        self.elem_proxfun = elem_proxfun

    def eval(self, x, scaling = 1.0):
        return scaling * self.elem_proxfun.eval(np.linalg.norm(x) / scaling)

    def eval_conjugate(self, x, scaling = 1.0):
        return scaling * self.elem_proxfun.eval_conjugate(np.linalg.norm(x))

    def eval_grad(self, x, scaling = 1.0):
        if np.linalg.norm(x) == 0.:
            return 0 * x
        return self.elem_proxfun.eval_grad(np.linalg.norm(x) / scaling) * x / np.linalg.norm(x)

    def eval_grad_conjugate(self, x, scaling = 1.0):
        if np.linalg.norm(x) == 0.:
            return 0 * x
        return scaling * self.elem_proxfun.eval_grad_conjugate(np.linalg.norm(x)) * x / np.linalg.norm(x)


class AnisotropicProxFunction(ProxFunction):
    def __init__(self, elem_proxfun):
        self.elem_proxfun = elem_proxfun

    def eval(self, x, scaling=1.0):
        return scaling * np.sum(self.elem_proxfun.eval(x / scaling))

    def eval_conjugate(self, x, scaling=1.0):
        return scaling * np.sum(self.elem_proxfun.eval_conjuate(x))

    def eval_grad(self, x, scaling=1.0):
        return self.elem_proxfun.eval_grad(x / scaling)

    def eval_grad_conjugate(self, x, scaling=1.0):
        return scaling * self.elem_proxfun.eval_grad_conjugate(x)


class Proxable(ABC):
    def __init__(self, proxable):
        self.proxable = proxable
        self.tau = 1

    def eval(self, x):
        return self.proxable.eval(x)

    def eval_isotropic_prox(self, y, v, proxfun, scaling = 1.):
        def residual(tau):
            z = np.linalg.norm(self.proxable.eval_prox(y + 1. / tau * v, 1. / tau) - y)

            return tau - (
                proxfun.elem_proxfun.eval_grad(z / scaling)
            ) / z

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

        for k in range(300):
            self.tau = tau_low + (tau_high - tau_low) / 2
            res = residual(self.tau)
            # if residual(self.tau) < 1e-15:
            #    break

            if res < 0:
                tau_low = self.tau
            elif res > 0:
                tau_high = self.tau
            else:
                break

        return self.proxable.eval_prox(y + 1. / self.tau * v, 1 / self.tau)

    def eval_prox(self, y, v, proxfun, scaling = 1.):
        if isinstance(proxfun.elem_proxfun, PowerElemProxFunction) and proxfun.elem_proxfun.p == 2:
            return self.proxable.eval_prox(y + scaling * v, scaling)

        if isinstance(proxfun, AnisotropicProxFunction):
            return self.eval_anisotropic_prox(y, v, proxfun, scaling)
        elif isinstance(proxfun, IsotropicProxFunction):
            return self.eval_isotropic_prox(y, v, proxfun, scaling)
        else:
            return None

    @abstractmethod
    def eval_anisotropic_prox(self, y, v, proxfun, scaling = 1.):
        pass

class IndicatorBox(Proxable):
    def __init__(self):
        super().__init__(fun.IndicatorBox())

    def eval_anisotropic_prox(self, y, v, proxfun, scaling=1.):
        return np.minimum(1., np.maximum(-1., y + proxfun.eval_grad_conjugate(v, scaling)))


class IndicatorSimplex(Proxable):
    def __init__(self):
        super().__init__(fun.IndicatorSimplex())
    ##
    # minimize_{x in Delta} lamb phi((x - y) / lamb) - <x,b>
    ##
    def eval_anisotropic_prox(self, y, b, proxfun, lamb):
        #implements bisection to find multiplier eta for dualizing the sum-to-one constraint
        #min_{x >=0 } max_eta phi(x - y) - <x, b> + eta * (<1, x> - 1)
        #returns [y - nabla phi^*(eta - b)]_+

        #first step identify relevant piece of the dual function g(eta)
        n = y.shape[0]
        w = proxfun.eval_grad(y, lamb) + b
        idx_sorted = np.argsort(w)
        w_sorted = w[idx_sorted]

        deriv = lambda eta, j: -np.sum(proxfun.eval_grad_conjugate(eta - b[idx_sorted[j:]], lamb)) + np.sum(y[idx_sorted[j:]]) - 1
        eta_minus = 0
        for j in range(n):
            eta_minus = w_sorted[j]
            if deriv(eta_minus, j) <= 0:
                break

        if j == 0:
            eta_plus = w_sorted[0]
            while True:
                eta_plus = eta_plus - 10.
                if deriv(eta_plus, j) > 0:
                    break
        else:
            eta_plus = w_sorted[j - 1]

        # bisection
        eta = 0.
        for i in range(500):
            eta = eta_plus + (eta_minus - eta_plus) / 2

            v = deriv(eta, j)
            if np.abs(v) < 1e-15:
                break

            #print(i, eta_minus, eta_plus, v, midpoint)

            if v > 0:
                eta_plus = eta
            elif v < 0:
                eta_minus = eta
            else:
                break

        return np.maximum(0., y - proxfun.eval_grad_conjugate(eta - b, lamb))



