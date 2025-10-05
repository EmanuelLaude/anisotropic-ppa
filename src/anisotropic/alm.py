import numpy as np
import compot.calculus.function as fun
import compot.optimizer.base as base
import compot.optimizer.classic as cls

from abc import abstractmethod

class SaddlePointProblem:
    def __init__(self, x_init, y_init, A, b, proxable_primal, diffable_primal, proxable_dual):
        self.x_init = x_init
        self.y_init = y_init
        self.A = A
        self.b = b

        self.proxable_primal = proxable_primal
        self.proxable_dual = proxable_dual
        self.diffable_primal = diffable_primal

    @abstractmethod
    def eval_primal(self, x):
        pass

    @abstractmethod
    def eval_dual(self, y):
        pass


#min_x f(Ax-b) + g(x)
#min_x max_y <Ax - b, y> + g(x) - f^*(y)
#L_\lamb(x,y) \sup_{\eta} <Ax - b,\eta> -lambda * \phi(\eta - y) + g(x) - f^*(x)
class AugmentedLagrangianFunction(fun.Diffable):
    def __init__(self, x, y, problem, proxfun_primal, proxfun_dual, tau, sigma, oracle, params_oracle, maxit_cumsum_inner, bounds=None):
        #primal
        self.x = x
        self.tau = tau
        self.proxfun_primal = proxfun_primal

        #dual
        self.y = y
        self.sigma = sigma
        self.bounds = bounds
        self.proxfun_dual = proxfun_dual

        self.problem = problem
        self.oracle = oracle
        self.maxit_cumsum_inner = maxit_cumsum_inner
        self.params_oracle = params_oracle

        self.it_cumsum_inner = 0

    def eval_gradient(self, x):
        v = self.problem.A.apply(x) - self.problem.b

        y = self.problem.proxable_dual.eval_prox(self.y, v, self.proxfun_dual, self.sigma)

        return (self.problem.A.apply_transpose(y) + self.problem.diffable_primal.eval_gradient(x)
                + (self.proxfun_primal.eval_grad(x - self.x, self.tau) if self.tau != np.inf else 0.))

    def eval(self, x):
        v = self.problem.A.apply(x)

        y = self.problem.proxable_dual.eval_prox(self.y, v - self.problem.b, self.proxfun_dual, self.sigma)

        return (np.dot(v - self.problem.b, y) + self.problem.diffable_primal.eval(x) - self.proxfun_dual.eval(y - self.y, self.sigma)
                + (self.proxfun_primal.eval(x - self.x, self.tau) if self.tau != np.inf else 0.))

    def update_primal(self, tol):
        if self.bounds is None:
            proxable_primal = self.problem.proxable_primal
        else:
            lb = self.x-self.bounds
            ub = self.x+self.bounds
            if isinstance(self.problem.proxable_primal, fun.IndicatorBox):
                lb = np.maximum(lb, self.problem.proxable_primal._l)
                ub = np.minimum(ub, self.problem.proxable_primal._u)
            proxable_primal = fun.IndicatorBox(l=lb, u=ub)

        problem = base.CompositeOptimizationProblem(self.x, self, proxable_primal)
        def callback(x, status):
            if status.nit > 0:
                self.it_cumsum_inner += 1
            print("    ", status.nit, self.it_cumsum_inner, problem.eval_objective(x), status.fres)

            if self.it_cumsum_inner >= self.maxit_cumsum_inner:
                return True
            return False

        self.params_oracle.ftol = tol
        self.params_oracle.gtol = tol
        optimizer = self.oracle(self.params_oracle, problem, callback=callback)
        status = optimizer.run()
        self.x[:] = optimizer.x[:]
        return status

    def update_dual(self):
        v = self.problem.A.apply(self.x) - self.problem.b

        self.y[:] = self.problem.proxable_dual.eval_prox(self.y, v, self.proxfun_dual, self.sigma)[:]

class Status(base.Status):
    def __init__(self, nit=0, res=np.inf, success=False, eps=1e-13, status_oracle=base.Status(), cumsum_iters_inner=0):
        super().__init__(nit, res, success)
        self.eps = eps
        self.status_oracle = status_oracle
        self.cumsum_iters_inner = cumsum_iters_inner


class Parameters(base.Parameters):
    def __init__(self, maxit=500, tol=1e-5, epsilon=1e-12, tau = 1.0, sigma = 1.0, class_oracle = cls.LBFGS,
                 params_oracle = cls.Parameters(), maxit_cumsum_inner = 20000, proxfun_primal = None, proxfun_dual = None, stopping_criterion = None, tolerance=None):
        super().__init__(maxit, tol, epsilon)

        self.tau = tau
        self.sigma = sigma
        self.class_oracle = class_oracle
        self.params_oracle = params_oracle
        self.proxfun_primal = proxfun_primal
        self.proxfun_dual = proxfun_dual
        self.stopping_criterion = stopping_criterion
        self.tolerance = tolerance
        self.maxit_cumsum_inner = maxit_cumsum_inner

class AugmentedLagrangianMethod(base.PrimalDualIterativeOptimizer):
    def __init__(self, params, problem, callback):
        super().__init__(params, problem, self.callback)
        self._callback = callback
        self.status = Status()

    def callback(self, x, status):
        result = self._callback(x, status)
        if self.status.cumsum_iters_inner >= self.params.maxit_cumsum_inner:
            return True

        return result

    def setup(self):
        self.augmented_lagrangian = AugmentedLagrangianFunction(self.x, self.y, self.problem,
                                                                self.params.proxfun_primal,
                                                                self.params.proxfun_dual,
                                                                self.params.tau,
                                                                self.params.sigma,
                                                                self.params.class_oracle,
                                                                self.params.params_oracle,
                                                                self.params.maxit_cumsum_inner,
                                                                self.params.bounds)


    def pre_step(self, _):
        return np.inf

    def step(self, k):
        tol = self.params.tolerance(k)
        self.status.status_oracle = self.augmented_lagrangian.update_primal(tol)

        self.status.cumsum_iters_inner += self.status.status_oracle.nit

        self.augmented_lagrangian.update_dual()

