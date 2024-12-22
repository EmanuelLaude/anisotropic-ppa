import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import benchmarks
import compot.calculus.function as fun
import compot.optimizer.lipschitz as opt_lip

import anisotropic.function as prox
import anisotropic.alm as aalm

class ZeroSumGame(aalm.SaddlePointProblem):
    def __init__(self, x0, y0, A):
        super().__init__(x0, y0, A, np.zeros(A.shape[0]),
                                            fun.IndicatorSimplex(),
                                            fun.Zero(),
                                            prox.IndicatorSimplex()
        )
        self.x0 = self.proxable_primal.eval_prox(x0, 1)

        self.y0 = self.proxable_dual.proxable.eval_prox(y0, 1)


    def eval_dual(self, y):
        v = np.dot(-self.A.T, y)

        return -np.max(v)

    def eval_primal(self, x):
        return np.max(np.dot(self.A, x))


class ZeroSumGameBenchmark(benchmarks.Benchmark):
    def setup(self):
        m, n = self.config["m"], self.config["n"]
        self.A = 10 * np.random.rand(m, n) - 5

        self.x0 = np.random.randn(n)
        self.y0 = np.random.randn(m)

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
        return ZeroSumGame(x_init, y_init, self.A)

    def get_solution(self):
        problem = ZeroSumGame(np.zeros(self.dim_primal), np.zeros(self.dim_dual), self.A)
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

            print("nit", i, "k", k, "|Ax|", np.linalg.norm(v, 1), "primal", primal, "dual", dual, "gap", primal - dual, "res", res)

        proxfun_dual = prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        )
        proxfun_primal = prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        )

        lamb = 100
        sigma = np.Inf

        params = opt_lip.Parameters()
        params.maxit = 1200
        params.tol = 1e-15
        params.epsilon = 1e-15
        params.Wolfe = True
        params.mem = 200

        solver = opt_lip.LBFGSPanoc
        maxit = 25

        def stopping_criterion(problem, x, k):
            return np.linalg.norm(problem.diffable.eval_gradient(x)) < 1e-15

        method = aalm.AugmentedLagrangianMethod(maxit, callback, solver, params, problem,
                                                lamb, sigma, proxfun_primal, proxfun_dual, stopping_criterion)

        method.run()

        return self.fmin_primal, self.fmax_dual, self.xopt, self.yopt



name = "zerosumgame"
num_runs = 1
config = {
        "name": "zerosum game",
        "n": 150,
        "m": 160,
        "markevery": 1,
        "plotevery": 20,
        "seed": 3201,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "ftol_primal": 1e-15,
        "ftol_dual": 1e-15,
        "tol_pdgap": 5e-11
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


maxit = 40
solver_class = opt_lip.LBFGSPanoc
params_solver = opt_lip.Parameters()
params_solver.gamma_init = 10
params_solver.maxit = 500
params_solver.tol = 1e-10
params_solver.epsilon = 1e-10
params_solver.mem = 20
params_solver.Wolfe = True

#def stopping_criterion(problem, x, k, p):
#    return np.linalg.norm(problem.diffable.eval_gradient(x)) < 1e-12 / ((k + 1) ** p)

optimizer_configs = [
    {
        "marker": "p",
        "color": "black",
        "linestyle": "dashdot",
        "name": "alm_p4_u2_lamb1_sigma1",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$p=4,u=2,\\lambda=1$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.AnisotropicProxFunction(
            prox.PowerElemProxFunction(4)
           #prox.SymmetricExpElemProxFunction(1.)
        ),
        "proxfun_dual": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "stopping_criterion": lambda problem, x, k: False,
        "lamb": 1,
        "sigma": 1,
        "markevery": 1,
        "plotevery": 20
    },
    {
        "marker": "*",
        "color": "black",
        "linestyle": "dashed",
        "name": "alm_exp_u2_lamb1_sigma1",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$\\exp, u=2,\\lambda=1$",
        "maxit": maxit,
        "solver": solver_class,
        "params_solver": params_solver,
        "proxfun_primal": prox.AnisotropicProxFunction(
            #prox.PowerElemProxFunction(4)
            prox.SymmetricExpElemProxFunction(0.01)
        ),
        "proxfun_dual": prox.IsotropicProxFunction(
            prox.PowerElemProxFunction(2)
        ),
        "stopping_criterion": lambda problem, x, k: False,
        "lamb": 1,
        "sigma": 1,
        "markevery": 1,
        "plotevery": 20
    },
    {
        "marker": "x",
        "color": "black",
        "linestyle": "solid",
        "name": "alm_p2_u2_lamb1_sigma1",
        "class": aalm.AugmentedLagrangianMethod,
        "label": "$p=2,u=2,\\lambda=1$",
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
        "stopping_criterion": lambda problem, x, k: False,
        "sigma": 1,
        "markevery": 1,
        "plotevery": 20
    }
]

zerosumgame = ZeroSumGameBenchmark(name, config, optimizer_configs, num_runs)
zerosumgame.run(overwrite_file=False)


matplotlib.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(4.5, 4))
#fig.suptitle(config["name"], fontsize=12)
ax.grid(True)
zerosumgame.plot_primal_dual_gap_cumsum(markevery=config["markevery"], plotevery=config["plotevery"])

plt.show()

# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# fig, ax = plt.subplots(figsize=(4.5, 4))
# fig.suptitle(config["name"], fontsize=12)
# ax.grid(True)
# zerosumgame.plot_suboptimality_primal(markevery=config["markevery"], plotevery=config["plotevery"])
# plt.show()

