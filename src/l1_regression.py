import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import benchmarks
import compot.calculus.function as fun
import compot.optimizer.base as opt_base
import compot.optimizer.classic as opt_cls

import anisotropic.function as prox
import anisotropic.alm as aalm

class L1RegressionProblem(aalm.SaddlePointProblem):
    def __init__(self, x0, y0, A, b, theta):
        super().__init__(x0, y0, A, b,
                                            fun.Zero(),
                                            fun.FunctionTransform(
                                                fun.NormPower(power=2, norm=2),
                                                rho=theta
                                            ),
                                            prox.IndicatorBox()
        )
        self.theta = theta

    def eval_dual(self, y):
        v = np.dot(-self.A.T, y)

        return (-0.5 * np.dot(v, v) / self.theta if self.theta > 0 else 0.) - np.dot(self.b, y)

    def eval_primal(self, x):
        return np.sum(np.abs(np.dot(self.A, x) - self.b)) + 0.5 * self.theta * np.dot(x, x)



class L1Regression(benchmarks.Benchmark):
    def setup(self):
        m, n = self.config["m"], self.config["n"]
        self.A = 10 * np.random.rand(m, n) - 5
        self.b = np.random.randn(m)

        self.x0 = np.random.randn(n)
        self.y0 = np.random.randn(m)

        self.theta = self.config["theta"]

        self.fmin_primal = np.Inf
        self.fmax_dual = -np.Inf
        self.xopt = None
        self.yopt = None

        return n, m

    def get_filename(self):
        return ("results/" + self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"]))


    def setup_problem(self, x_init, y_init):
        return L1RegressionProblem(x_init, y_init, self.A, self.b, self.theta)

    def get_solution(self):
        problem = L1RegressionProblem(np.zeros(self.dim_primal), np.zeros(self.dim_dual), self.A, self.b, self.theta)
        def callback(i, k, x, y, res):
            v = np.dot(self.A, x)

            primal = problem.eval_primal(x)
            if primal < self.fmin_primal:
                self.fmin_primal = primal
                self.xopt = x

            dual = problem.eval_dual(y)
            if dual > self.fmax_dual:
                self.fmax_dual = dual
                self.yopt = y

            print("nit", i, "k", k, "|Ax - b|", np.linalg.norm(v - self.b, 1), "primal", primal, "dual", dual, "gap", primal - dual,
                  "grad al", np.linalg.norm(method.augmented_lagrangian.eval_gradient(x)), "res", res)

        proxfun_dual = prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        )
        proxfun_primal = prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        )

        lamb = 100
        sigma = np.Inf

        params = opt_cls.Parameters()
        params.maxit = 800
        params.tol = 1e-13
        params.epsilon = 1e-10
        params.Wolfe = True
        params.mem = 200

        solver = opt_cls.LBFGS
        maxit = 10

        def stopping_criterion(problem, x, k):
            return np.linalg.norm(problem.diffable.eval_gradient(x)) < 1e-10 / ((k + 1) ** 2)

        method = aalm.AugmentedLagrangianMethod(maxit, callback, solver, params, problem,
                                                lamb, sigma, proxfun_primal, proxfun_dual, stopping_criterion)

        method.run()

        return self.fmin_primal, self.fmax_dual, self.xopt, self.yopt



name = "l1_regression"
num_runs = 1
config = {
        "name": "$\\ell_1$-regression",
        "n": 145,
        "m": 150,
        "markevery": 1,
        "plotevery": 20,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "theta": 0.1,
        "ftol_primal": 1e-9,
        "ftol_dual": 1e-15,
        "tol_pdgap": 1e-15
    }
# config = {
#         "name": "l1_regression400",
#         "n": 145,
#         "m": 120,
#         "markevery": 1,
#         "plotevery": 20,
#         "seed": 120,
#         "init_proc": "np.random.randn",
#         "verbose": 1,
#         "theta": 100,
#         "ftol_primal": 1e-15,
#         "ftol_dual": 1e-15
#     }


maxit = 20
solver_class = opt_base.ScipyBFGSWrapper
params_solver = opt_base.Parameters()
params_solver.maxit = 800
params_solver.tol = 1e-12
params_solver.epsilon = 1e-8
params_solver.mem = 10
params_solver.Wolfe = True

def stopping_criterion(problem, x, k, p):
    return False#np.linalg.norm(problem.diffable.eval_gradient(x)) < 1e-12 / ((k + 1) ** p)

optimizer_configs = [
    {
        "marker": "^",
        "color": "black",
        "linestyle": "dashdot",
        "name": "alm_p4_lamb2_sigma2",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$p=4,\\lambda=2$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.AnisotropicProxFunction(
            prox.PowerElemProxFunction(4)
        ),
        "proxfun_dual": prox.AnisotropicProxFunction(
            prox.PowerElemProxFunction(4)
        ),
        "stopping_criterion": lambda problem, x, k: stopping_criterion(problem, x, k, 4),
        "lamb": 2,
        "sigma": 2,
        "markevery": 1,
        "plotevery": 20,
    },
    {
        "marker": "*",
        "color": "black",
        "linestyle": "dashed",
        "name": "alm_cosh_lamb1_sigma1",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$\\cosh,\\lambda=1$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.AnisotropicProxFunction(
            #prox.PowerElemProxFunction(4)
            #prox.SymmetricExpElemProxFunction(0.01)
            prox.CoshElemProxFunction()
        ),
        "proxfun_dual": prox.AnisotropicProxFunction(
            #prox.PowerElemProxFunction(4)
            #prox.SymmetricExpElemProxFunction(0.01)
            prox.CoshElemProxFunction()
        ),
        "stopping_criterion": lambda problem, x, k: stopping_criterion(problem, x, k, 2),
        "lamb": 1,
        "sigma": 1,
        "markevery": 1,
        "plotevery": 20
    },
    {
        "marker": "o",
        "color": "black",
        "linestyle": "solid",
        "name": "alm_p2_lamb100_sigma100",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$p=2,\\lambda=100$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "proxfun_dual": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "stopping_criterion": lambda problem, x, k: stopping_criterion(problem, x, k, 2),
        "lamb": 100,
        "sigma": 100,
        "markevery": 1,
        "plotevery": 20
    },
    {
        "marker": "X",
        "color": "black",
        "linestyle": "solid",
        "name": "Prox-Power-ALM2_10",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$p=2,\\lambda=10$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "proxfun_dual": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "lamb": 10,
        "stopping_criterion": lambda problem, x, k: stopping_criterion(problem, x, k, 2),
        "sigma": 10,
        "markevery": 1,
        "plotevery": 20
    },
    {
        "marker": "p",
        "color": "black",
        "linestyle": "solid",
        "name": "Prox-Power-ALM2_1",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$p=2,\\lambda=1$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "proxfun_dual": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "lamb": 1,
        "stopping_criterion": lambda problem, x, k: stopping_criterion(problem, x, k, 2),
        "sigma": 1,
        "markevery": 1,
        "plotevery": 20
    }
]

l1_regression = L1Regression(name, config, optimizer_configs, num_runs)
l1_regression.run(overwrite_file=False)

label_fontsize = 'scriptsize'

# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# fig, ax = plt.subplots(figsize=(4.5, 4))
# fig.suptitle(config["name"], fontsize=12)
# ax.grid(True)
# l1_regression.plot_primal_dual_gap_cumsum(markevery=config["markevery"], plotevery=config["plotevery"])
# plt.show()

matplotlib.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(4.5, 4))
#fig.suptitle(config["name"], fontsize=12)
ax.grid(True)
l1_regression.plot_suboptimality_primal_cumsum(markevery=config["markevery"], plotevery=config["plotevery"])
plt.show()


