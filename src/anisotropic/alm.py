import numpy as np
import compot.calculus.function as fun
import compot.optimizer.base as opt_base

from abc import ABC, abstractmethod


class SaddlePointProblem:
    def __init__(self, x0, y0, A, b, proxable_primal, diffable_primal, proxable_dual):
        self.x0 = x0
        self.y0 = y0
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
class AugmentedLagrangian(fun.Diffable):
    def __init__(self, x, y, problem, proxfun_primal, proxfun_dual, lamb, sigma, solver, params, stopping_criterion):
        #primal
        self.x = x
        self.sigma = sigma
        self.proxfun_primal = proxfun_primal

        #dual
        self.y = y
        self.lamb = lamb
        self.proxfun_dual = proxfun_dual

        self.problem = problem
        self.solver = solver
        self.stopping_criterion = stopping_criterion
        self.params = params

        self.iters_inner = 0

    def eval_gradient(self, x):
        v = np.dot(self.problem.A, x)

        y = self.problem.proxable_dual.eval_prox(self.y, v - self.problem.b, self.proxfun_dual, self.lamb)

        return (np.dot(self.problem.A.T, y) + self.problem.diffable_primal.eval_gradient(x)
                + (self.proxfun_primal.eval_grad(x - self.x, self.sigma) if self.sigma != np.Inf else 0.))

    def eval(self, x):
        v = np.dot(self.problem.A, x)

        y = self.problem.proxable_dual.eval_prox(self.y, v - self.problem.b, self.proxfun_dual, self.lamb)

        return (np.dot(v - self.problem.b, y) + self.problem.diffable_primal.eval(x) - self.proxfun_dual.eval(y - self.y, self.lamb)
                + (self.proxfun_primal.eval(x - self.x, self.sigma) if self.sigma != np.Inf else 0.))

    def update_primal(self):
        problem = opt_base.CompositeOptimizationProblem(self.x, self, self.problem.proxable_primal)
        def callback(x, status):
            #print("    ", status.nit, problem.eval_objective(x), np.linalg.norm(problem.diffable.eval_gradient(x)), status.res, status.tau)

            return self.stopping_criterion(problem, x, status.nit)


        optimizer = self.solver(self.params, problem, callback=callback)
        status = optimizer.run()
        self.x[:] = optimizer.x[:]
        return status

    def update_dual(self):
        v = np.dot(self.problem.A, self.x)

        self.y[:] = self.problem.proxable_dual.eval_prox(self.y, v - self.problem.b, self.proxfun_dual, self.lamb)[:]



class AugmentedLagrangianMethod:
    def __init__(self, maxit, callback, solver, params, problem, lamb, sigma, proxfun_primal, proxfun_dual, stopping_criterion):

        self.x = np.copy(problem.x0)
        self.y = np.copy(problem.y0)
        self.maxit = maxit
        self.cumsum_iters_inner = 0
        self.callback = callback
        self.solver = solver

        self.augmented_lagrangian = AugmentedLagrangian(self.x, self.y, problem,
                 proxfun_primal, proxfun_dual, lamb, sigma, solver, params, stopping_criterion)


    def run(self):
        res = np.Inf
        for i in range(self.maxit):
            if self.callback(i, self.cumsum_iters_inner, self.x, self.y, res):
                break
            status = self.augmented_lagrangian.update_primal()

            self.cumsum_iters_inner += status.nit
            res = status.res

            self.augmented_lagrangian.update_dual()
