from abc import ABC, abstractmethod

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from math import floor, log10

def fexp10(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman10(f):
    return f / 10 ** fexp10(f)



class Benchmark(ABC):
    def __init__(self, name, config, optimizer_configs, num_runs):
        self.name = name
        self.config = config
        self.optimizer_configs = optimizer_configs
        self.num_runs = num_runs

        np.random.seed(self.config["seed"])

        self.dim_primal, self.dim_dual = self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_solution(self):
        pass

    @abstractmethod
    def get_filename(self):
        pass

    @abstractmethod
    def setup_problem(self, x_init, y_init):
        pass

    def run(self, overwrite_file=False):
        filename = self.get_filename()
        suffix = ".npz"

        if overwrite_file or not Path(filename + suffix).is_file():
            (self.fmin_primal, self.fmax_dual, self.xopt, self.yopt) = self.get_solution()

            self.primal_objective = dict()
            self.dual_objective = dict()
            self.dist_primal = dict()
            self.dist_dual = dict()
            self.cumsum_iters = dict()
        else:
            cache = np.load(filename + '.npz', allow_pickle=True)
            self.fmin_primal = cache["fmin_primal"]
            self.xopt = cache["xopt"]
            self.fmax_dual = cache["fmax_dual"]
            self.yopt = cache["yopt"]

            self.primal_objective = cache["primal_objective"].item()
            self.dual_objective = cache["dual_objective"].item()
            self.dist_primal = cache["dist_primal"].item()
            self.dist_dual = cache["dist_dual"].item()
            self.cumsum_iters = cache["cumsum_iters"].item()


        update_file = False

        for optimizer_config in self.optimizer_configs:
            if optimizer_config["name"] in self.primal_objective:
                continue

            update_file = True
            np.random.seed(self.config["seed"])

            self.primal_objective[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.dual_objective[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.dist_primal[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.dist_dual[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.cumsum_iters[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]

            for run in range(self.num_runs):
                x_init = eval(self.config["init_proc"] + "(" + str(self.dim_primal) + ")")
                y_init = eval(self.config["init_proc"] + "(" + str(self.dim_dual) + ")")
                problem = self.setup_problem(x_init, y_init)


                def callback(i, k, x, y, res):
                    primal_value = problem.eval_primal(x)
                    dual_value = problem.eval_dual(y)

                    self.dist_primal[optimizer_config["name"]][run].append(np.linalg.norm(x - self.xopt))
                    self.dist_dual[optimizer_config["name"]][run].append(np.linalg.norm(y - self.yopt))

                    self.primal_objective[optimizer_config["name"]][run].append(primal_value)
                    self.dual_objective[optimizer_config["name"]][run].append(dual_value)

                    self.cumsum_iters[optimizer_config["name"]][run].append(k)

                    if i % self.config["verbose"] == 0:
                        print("    nit", i, "cumsum iters", k, "primal", primal_value - self.fmin_primal, "dual", dual_value - self.fmax_dual, "gap", primal_value - dual_value, "res", res)

                    if abs(primal_value - self.fmin_primal) < self.config["ftol_primal"]:
                        return True

                    if abs(dual_value - self.fmax_dual) < self.config["ftol_dual"]:
                        return True

                    if abs(primal_value - dual_value) < self.config["tol_pdgap"]:
                        return True

                    return False

                optimizer = optimizer_config["class"](optimizer_config["maxit"],
                                                      callback,
                                                      optimizer_config["solver"],
                                                      optimizer_config["params_solver"],
                                                      problem,
                                                      optimizer_config["lamb"],
                                                      optimizer_config["sigma"],
                                                      optimizer_config["proxfun_primal"],
                                                      optimizer_config["proxfun_dual"],
                                                      optimizer_config["stopping_criterion"])

                print(optimizer_config["name"])

                optimizer.run()

        if update_file:
            np.savez(filename,
                     fmin_primal=self.fmin_primal,
                     fmax_dual=self.fmax_dual,
                     xopt=self.xopt,
                     yopt=self.yopt,
                     primal_objective=self.primal_objective,
                     dual_objective=self.dual_objective,
                     dist_primal=self.dist_primal,
                     dist_dual=self.dist_dual,
                     cumsum_iters=self.cumsum_iters
                     )


    def linspace_values(self, x, y, interval):
        values = np.arange(0, x[-1], interval) * 0.

        j = 0
        for i in range(0, x[-1], interval):
            while True:
                if x[j] > i:
                    break
                j = j + 1

            # linearly interpolate the values at j-1 and j to obtain the value at i
            values[int(i / interval)] = (
                    y[j - 1]
                    + (i - x[j - 1])
                    * (y[j] - y[j - 1]) / (x[j] - x[j - 1])
            )
        return values

    def plot_mean_stdev(self, xvals, yvals, label, marker, color, refval=0., plotstdev=True, markevery=20,
                        plotevery=250):

        # compute new array with linspaced xvals with shortest length
        xvals_linspace = np.arange(0, xvals[0][-1], plotevery)
        for i in range(1, len(xvals)):
            arange = np.arange(0, xvals[i][-1], plotevery)
            if len(xvals_linspace) > len(arange):
                xvals_linspace = arange

        yvals_mean = np.zeros(len(xvals_linspace))

        for i in range(len(xvals)):
            y_values_interp = self.linspace_values(xvals[i],
                                                   yvals[i], plotevery)
            yvals_mean += y_values_interp[0:len(xvals_linspace)]

        yvals_mean = yvals_mean / len(xvals)

        plt.semilogy(xvals_linspace, yvals_mean - refval,
                     label=label,
                     marker=marker,
                     markevery=markevery,
                     color=color)

        if len(xvals) > 1 and plotstdev:
            yvals_stdev = np.zeros(len(xvals_linspace))

            for i in range(len(xvals)):
                y_values_interp = self.linspace_values(xvals[i],
                                                       yvals[i], plotevery)

                yvals_stdev += (yvals_mean - y_values_interp[0:len(xvals_linspace)]) ** 2

            yvals_stdev = np.sqrt(yvals_stdev / len(xvals))

            plt.fill_between(xvals_linspace,
                             yvals_mean - refval - yvals_stdev,
                             yvals_mean - refval + yvals_stdev,
                             alpha=0.5, facecolor=color,
                             edgecolor='white')

    def plot_primal_dual_gap_cumsum(self, markevery, plotevery):
        gap = dict()
        for optimizer_config in self.optimizer_configs:
            gap[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            for i in range(self.num_runs):
                gap[optimizer_config["name"]][i] = np.array(self.primal_objective[optimizer_config["name"]][i]) - \
                                                   np.array(self.dual_objective[optimizer_config["name"]][i])

        self.plot(self.cumsum_iters, gap, 0., "cumulative number of inner iterations", "gap", markevery, plotevery)

    def plot_primal_dual_gap(self, markevery, plotevery):
        gap = dict()
        for optimizer_config in self.optimizer_configs:
            gap[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            for i in range(self.num_runs):
                gap[optimizer_config["name"]][i] = np.array(self.primal_objective[optimizer_config["name"]][i]) - \
                                                   np.array(self.dual_objective[optimizer_config["name"]][i])

        iters = dict()
        for optimizer_config in self.optimizer_configs:
            iters[optimizer_config["name"]] = [[i for i in range(len(self.primal_objective[optimizer_config["name"]][0]))] for _ in range(self.num_runs)]



        self.plot(iters, gap, 0., "iterations", "gap", markevery, plotevery)

    def plot_suboptimality_primal_cumsum(self, markevery, plotevery):
        self.plot(self.cumsum_iters, self.primal_objective, self.fmin_primal, "cumulative number of inner iterations", "suboptimality", markevery, plotevery)

    def plot_suboptimality_primal(self, markevery, plotevery):
        iters = dict()
        for optimizer_config in self.optimizer_configs:
            iters[optimizer_config["name"]] = [[i for i in range(len(self.primal_objective[optimizer_config["name"]][0]))] for _ in range(self.num_runs)]


        self.plot(iters, self.primal_objective, self.fmin_primal, "iterations",
                  "suboptimality", markevery, plotevery)

    def plot_suboptimality_dual_cumsum(self, markevery, plotevery):
        negative_dual = dict()
        for optimizer_config in self.optimizer_configs:
            negative_dual[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            for i in range(self.num_runs):
                negative_dual[optimizer_config["name"]][i] = -np.array(self.dual_objective[optimizer_config["name"]][i])

        self.plot(self.cumsum_iters, negative_dual, -self.fmax_dual, "cumulative number of inner iterations", "dual suboptimality", markevery, plotevery)

    def plot_suboptimality_dual(self, markevery, plotevery):
        negative_dual = dict()
        for optimizer_config in self.optimizer_configs:
            negative_dual[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            for i in range(self.num_runs):
                negative_dual[optimizer_config["name"]][i] = -np.array(self.dual_objective[optimizer_config["name"]][i])

        iters = dict()
        for optimizer_config in self.optimizer_configs:
            iters[optimizer_config["name"]] = [
                [i for i in range(len(self.primal_objective[optimizer_config["name"]][0]))] for _ in
                range(self.num_runs)]

        self.plot(iters, negative_dual, -self.fmax_dual, "iterations", "dual suboptimality", markevery, plotevery)

    def plot(self, xvals, yvals, refval, xlabel, ylabel, markevery, plotevery):
        for optimizer_config in self.optimizer_configs:
            if self.num_runs == 1:
                plt.semilogy(xvals[optimizer_config["name"]][0], np.array(yvals[optimizer_config["name"]][0]) - refval,
                             label=optimizer_config["label"],
                             marker=optimizer_config["marker"],
                             markevery=markevery,
                             color=optimizer_config["color"],
                             linestyle=optimizer_config["linestyle"],
                             )
            else:
                self.plot_mean_stdev(xvals, yvals[optimizer_config["name"]],
                                     label = optimizer_config["label"],
                                     marker = optimizer_config["marker"],
                                     color = optimizer_config["color"],
                                     refval = refval,
                                     plotstdev= True,
                                     markevery = markevery,
                                     plotevery = plotevery)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)


        plt.tight_layout()
        plt.legend()

