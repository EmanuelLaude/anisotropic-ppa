import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
import os

class Status:
    def __init__(self, nit = 0, res = np.Inf, res_resolvent = np.Inf):
        self.nit = nit
        self.res = res
        self.res_resolvent = res_resolvent

class Parameters:
    def __init__(self, maxit = 300, tol = 1e-12, type ='B'):
        self.maxit = maxit
        self.tol = tol
        self.type = type

class PowerProximalPointAlgorithm:
    def __init__(self, x_init, p, b, F, callback, params):
        self.x_init = np.copy(x_init)
        self.p = p
        self.q = p / (p - 1)
        self.F = F
        self.b = b
        self.callback = callback
        self.params = params

    def f(self, x):
        return np.sum(np.power(np.abs(x - self.b), self.p)) / self.p

    def fstar(self, x):
        return np.sum(np.power(np.abs(x), self.q)) / self.q + np.dot(self.b, x)

    def norm2(self, x, y):
        return np.linalg.norm(x - y)

    def normp(self, x, y):
        return np.power(np.sum(np.power(np.abs(x - y), self.p)), 1 / self.p)

    def normq(self, x, y):
        return np.power(np.sum(np.power(np.abs(x - y), self.q)), 1 / self.q)

    def S(self, x, p = None):
        if p is None:
            p = self.p
        return np.sign(x - self.b) * np.abs(np.power(x - self.b, p - 1))

    def D(self, x, y):
        return self.f(x) - self.f(y) - np.dot(self.S(y), x - y)

    def residual_resolvent(self, x, y, p):
        if self.params.type == 'B':
            return self.S(x, p) - self.S(y, p) + self.F(x)
        return self.S(x - y, p) + self.F(x)

    def resolvent(self, y):
        x = np.copy(y)
        for t in range(2, int(self.p) + 1, 1):
            x = fsolve(lambda x: self.residual_resolvent(x, y, t), x, xtol=1e-28)

        return x

    def run(self):
        x = np.copy(self.x_init)
        y = np.copy(self.x_init)
        self.status = Status()
        for i in range(self.params.maxit):
            self.status.nit = i
            self.status.res_resolvent = np.linalg.norm(self.residual_resolvent(x, y, self.p))
            self.status.res = np.linalg.norm(self.F(x))
            if self.callback(x, self.status):
                return x

            if self.status.res < self.params.tol:
                return x

            y = x
            x = self.resolvent(x)

        return x


class Runner:
    def __init__(self, configs, tol, sol, operator):
        self.configs = configs
        self.xvals = dict()
        self.yvals = dict()
        self.gap_bregman = dict()
        self.gap_euclidean = dict()
        self.gap_fstar = dict()
        self.gap_f = dict()
        self.tol = tol
        self.sol = sol
        self.operator = operator

    def run(self):
        for config in self.configs:
            params = Parameters(config["maxit"], self.tol, type=config["type"])

            self.xvals[config["id"]] = []
            self.yvals[config["id"]] = []
            self.gap_bregman[config["id"]] = []
            self.gap_euclidean[config["id"]] = []
            self.gap_fstar[config["id"]] = []
            self.gap_f[config["id"]] = []


            def callback(x, status):
                self.xvals[config["id"]].append(x[0])
                self.yvals[config["id"]].append(x[1])

                self.gap_bregman[config["id"]].append(power_ppa.D(self.sol, x))
                self.gap_euclidean[config["id"]].append(np.linalg.norm(x - self.sol))
                self.gap_fstar[config["id"]].append(power_ppa.normq(x, self.sol))
                self.gap_f[config["id"]].append(power_ppa.normp(x, self.sol))

                print("nit", status.nit, "res_resolvent", status.res_resolvent, "res", status.res, "x", x[0], "y", x[1])
                if np.linalg.norm(x - self.sol) <= 1e-3: # or power_ppa.D(self.sol, x) < self.tol:
                    return True

            power_ppa = PowerProximalPointAlgorithm(config["x_init"], config["p"], config["b"], self.operator, callback, params)
            power_ppa.run()

np.random.seed(1000)
#x_init = np.random.randn(2) * 25

theta = np.pi / 2
#A = 0.5 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#b = [1, 1]
b = np.array([0., 0.])
sol = np.linalg.solve(A, b)
x_init = np.random.randn(2) * 60
#x_init = sol + np.array([6, 6])
tol = 1e-16

def operator(x):
    return np.dot(A, x) - b


configs = [
    {
        "id": "anisotropic",
        "label": "anisotropic",
        "type": 'a',
        "p": 4,
        "b": np.zeros(2),
        "maxit": 500,
        "x_init": x_init
    },
]
runner = Runner(configs, tol, sol, operator)
runner.run()

label_fontsize = 'scriptsize'
axis_parameter_set = {
    'legend style={font=\\%s, legend cell align=left, align=left, draw=white!15!black}' % label_fontsize,
    'xticklabel style={font=\\%s}' % label_fontsize, 'yticklabel style={font=\\%s}' % label_fontsize}

prefix = "skew_symmetric_"
for config in configs:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plt.scatter(runner.xvals[config["id"]], runner.yvals[config["id"]], label='$x^k$ (' + config["label"] + ')', marker="o", color="black", s=1)
    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

linewidth = 2.5
markersize = 3.5
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.grid(True)

plt.semilogy(runner.gap_euclidean["anisotropic"][0:200], label='$\|x^k\|_2$', linewidth=linewidth, color="black", linestyle='dashed')
plt.semilogy(runner.gap_fstar["anisotropic"][0:200], label='$\|x^k\|_q$', linewidth=linewidth, color="black", linestyle='solid')
#plt.semilogy(runner.gap_f["anisotropic"][0:200], label='$\|x^k\|_p$', linewidth=linewidth, color="black", linestyle='dashdot')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.grid(True)

#plt.semilogy(runner.gap_euclidean["anisotropic"][0:200], label='$\|x^k\|_2$', linewidth=linewidth, color="black", linestyle='dashed')
plt.semilogy(runner.gap_fstar["anisotropic"][0:200], label='$\|x^k\|_q$', linewidth=linewidth, color="black", linestyle='solid')
plt.semilogy(runner.gap_f["anisotropic"][0:200], label='$\|x^k\|_p$', linewidth=linewidth, color="black", linestyle='dashdot')
plt.legend()
plt.show()