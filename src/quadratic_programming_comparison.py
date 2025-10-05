import matplotlib.pyplot as plt
import matplotlib

import compot.optimizer.scipy_wrapper as scipy_wrapper
import os

import quadratic_programming as qp

name = "quadprog"
num_runs = 1

class_oracle = scipy_wrapper.ScipyLBFGSBWrapper
maxit = 1e12
linewidth = 2.5
mem = 25
overwrite_file = True

configs = [
    {
        "name": "cont-050",
        "problem_name": "CONT-050",
        "markevery": 1,
        "plotevery": 1,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "max_cumsum_iters_inner": 15000,
        "eps": 1e-8,
        "maxit_per_step": 1000,
        "subopt_rel": 1e-4,
        "feas_rel": 1e-4,
        "optimizer_configs": [
            {
                "p": 3,
                "sigma": 1,
                "tau": 1e3,
            },
            {
                "p":2,
                "sigma": 1,
                "tau": 1e3,
            },
            {
                "p": 2,
                "sigma": 10,
                "tau": 1e5,
            },
            {
                "p":2,
                "sigma": 1e2,
                "tau": 1e5,
            },
        ]
    },
    {
        "name": "cont-100",
        "problem_name": "CONT-100",
        "markevery": 1,
        "plotevery": 1,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "maxit_per_step": 2000,
        "max_cumsum_iters_inner": 50000,
        "subopt_rel": 1e-5,
        "feas_rel": 1e-4,
        "eps": 1e-6,
        "optimizer_configs": [
            {
                "p": 3,
                "sigma": 10,
                "tau": 1e3,
            },
            {
                "p":2,
                "sigma": 10,
                "tau": 1e3,
            },
            {
                "p": 2,
                "sigma": 100,
                "tau": 1e5,
            },
            {
                "p":2,
                "sigma": 1e3,
                "tau": 1e5,
            },
        ]
    },
    {
        "name": "cvxqp1_m",
        "problem_name": "CVXQP1_M",
        "markevery": 1,
        "plotevery": 1,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "maxit_per_step": 800,
        "max_cumsum_iters_inner": 50000,
        "subopt_rel": 1e-6,
        "feas_rel": 1e-6,
        "eps": 1e-8,
        "optimizer_configs": [
            {
                "p": 3,
                "sigma": 1e5,
                "tau": 1e2
            },
            {
                "p": 2,
                "sigma": 1e5,
                "tau": 1e2
            },
            {
                "p": 2,
                "sigma": 1e8,
                "tau": 1e5,
            },
            {
                "p": 2,
                "sigma": 1e10,
                "tau": 1e5,
            }
        ]
    },
    {
        "name": "cvxqp2_s",
        "problem_name": "CVXQP2_S",
        "markevery": 1,
        "plotevery": 1,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "maxit_per_step": 100,
        "max_cumsum_iters_inner": 3000,
        "subopt_rel": 1e-6,
        "feas_rel": 1e-6,
        "eps": 1e-8,
        "optimizer_configs": [
            {
                "sigma": 700,
                "tau": 1e2,
                "p": 3
            },
            {
                "sigma": 700,
                "tau": 1e2,
                "p": 2
            },
            {
                "sigma": 1000,
                "tau": 1e5,
                "p": 2
            },
            {
                "sigma": 2000,
                "tau": 1e5,
                "p": 2
            },
        ]
    },
    {
        "name": "gouldqp2",
        "problem_name": "GOULDQP2",
        "markevery": 1,
        "plotevery": 1,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "maxit_per_step": 8,
        "max_cumsum_iters_inner": 2000,
        "subopt_rel": 1e-4,
        "feas_rel": 1e-5,
        "eps": 1e-5,
        "optimizer_configs": [
            {
                "sigma": 0.1,
                "tau": 1e5,
                "p": 3
            },
            {
                "sigma": 0.1,
                "tau": 1e5,
                "p": 2
            },
            {
                "sigma": 1,
                "tau": 1e5,
                "p": 2
            },
            {
                "sigma": 10,
                "tau": 1e5,
                "p": 2
            },
        ]
    },
    {
        "name": "mosarqp1",
        "problem_name": "MOSARQP1",
        "markevery": 1,
        "plotevery": 1,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "maxit_per_step": 120,
        "max_cumsum_iters_inner": 5000,
        "eps": 1e-4,
        "subopt_rel": 1e-6,
        "feas_rel": 1e-4,
        "optimizer_configs": [
            {
                "p": 3,
                "sigma": 1,
                "tau": 1e3,
            },
            {
                "p": 2,
                "sigma": 1,
                "tau": 1e3,
            },
            {
                "p": 2,
                "sigma": 10,
                "tau": 1e5,
            },
            {
                "p": 2,
                "sigma": 100,
                "tau": 1e5,
            },
        ]
    },
    {
        "name": "mosarqp2",
        "problem_name": "MOSARQP2",
        "markevery": 1,
        "plotevery": 20,
        "seed": 120,
        "init_proc": "np.random.randn",
        "verbose": 1,
        "maxit_per_step": 250,
        "max_cumsum_iters_inner": 12000,
        "subopt_rel": 1e-6,
        "feas_rel": 1e-6,
        "eps": 1e-8,
        "optimizer_configs": [
            {
                "sigma": 10,
                "tau": 1e3,
                "p": 3
            },
            {
                "sigma": 10,
                "tau": 1e3,
                "p": 2
            },
            {
                "sigma": 100,
                "tau": 1e5,
                "p": 2
            },
            {
                "sigma": 1000,
                "tau": 1e5,
                "p": 2
            },
        ]
    }
]

for config in configs:
    params_oracle = scipy_wrapper.Parameters()
    params_oracle.mem = mem
    params_oracle.tol = 0
    params_oracle.maxit = config["maxit_per_step"]

    optimizer_configs = qp.expand_optimizer_configs(
        config["optimizer_configs"],
        maxit,
        config["max_cumsum_iters_inner"],
        config["subopt_rel"],
        config["feas_rel"],
        class_oracle,
        params_oracle,
        config["eps"],
        linewidth
    )
    quad_prog = qp.QuadraticProgramming(name, config, optimizer_configs, num_runs)
    quad_prog.run(overwrite_file=overwrite_file)

print("Dataset & $n$ & $m$ & maxit total & maxit per step & $\\Delta_{\\mathrm{rel}}$ & $r_{\\mathrm{rel}}$ & $\\epsilon$")
for config in configs:
    params_oracle = scipy_wrapper.Parameters()
    params_oracle.mem = mem
    params_oracle.tol = 0
    params_oracle.maxit = config["maxit_per_step"]

    optimizer_configs = qp.expand_optimizer_configs(
        config["optimizer_configs"],
        maxit,
        config["max_cumsum_iters_inner"],
        config["subopt_rel"],
        config["feas_rel"],
        class_oracle,
        params_oracle,
        config["eps"],
        linewidth
    )
    quad_prog = qp.QuadraticProgramming(name, config, optimizer_configs, num_runs)
    quad_prog.run(overwrite_file=False)

    print(f"{config['problem_name']} & ${quad_prog.problem_data['qp']['Q'].shape[0]}$ & ${quad_prog.problem_data['qp']['A'].shape[0]}$ & ${config['max_cumsum_iters_inner']}$ & {params_oracle.maxit} & ${qp.num2tex(config['subopt_rel'])}$ & ${qp.num2tex(config['feas_rel'])}$ & ${qp.num2tex(config['eps'])}$")



print("Dataset & $p$ & $\\tau$ & $\\sigma$ & iters total & iters outer")
for config in configs:
    params_oracle = scipy_wrapper.Parameters()
    params_oracle.mem = mem
    params_oracle.tol = 0
    params_oracle.maxit = config["maxit_per_step"]

    optimizer_configs = qp.expand_optimizer_configs(
        config["optimizer_configs"],
        maxit,
        config["max_cumsum_iters_inner"],
        config["subopt_rel"],
        config["feas_rel"],
        class_oracle,
        params_oracle,
        config["eps"],
        linewidth
    )
    quad_prog = qp.QuadraticProgramming(name, config, optimizer_configs, num_runs)
    quad_prog.run(overwrite_file=False)

    for optimizer_config in optimizer_configs:
        iters_total = -1
        iters_outer = -1
        for i, (subopt, feas) in enumerate(zip(
            quad_prog.criteria["subopt_rel_cumsum_iters_inner"][optimizer_config["name"]][0]["yvals"],
            quad_prog.criteria["feas_rel_cumsum_iters_inner"][optimizer_config["name"]][0]["yvals"]
        )):
            if subopt <= optimizer_config["subopt_rel"] and feas <= optimizer_config["feas_rel"]:
                iters_total = quad_prog.criteria["subopt_rel_cumsum_iters_inner"][optimizer_config["name"]][0]["xvals"][i]
                iters_outer = i
                break

        print(f"{config['problem_name']} ${optimizer_config['p']}$ & ${qp.num2tex(optimizer_config['tau'])}$ & ${qp.num2tex(optimizer_config['sigma'])}$ & ${'--' if iters_total == -1 else qp.num2tex(iters_total)}$ & ${'--' if iters_outer == -1 else qp.num2tex(iters_outer)}$")


print("Dataset & $p$ & $\\tau$ & $\\sigma$ & $\\Delta_{\\mathrm{rel}}$ & $r_{\\mathrm{rel}}$")
for config in configs:
    params_oracle = scipy_wrapper.Parameters()
    params_oracle.mem = mem
    params_oracle.tol = 0
    params_oracle.maxit = config["maxit_per_step"]

    optimizer_configs = qp.expand_optimizer_configs(
        config["optimizer_configs"],
        maxit,
        config["max_cumsum_iters_inner"],
        config["subopt_rel"],
        config["feas_rel"],
        class_oracle,
        params_oracle,
        config["eps"],
        linewidth
    )
    quad_prog = qp.QuadraticProgramming(name, config, optimizer_configs, num_runs)
    quad_prog.run(overwrite_file=False)

    for optimizer_config in optimizer_configs:
        subopt_rel = quad_prog.criteria["subopt_rel_cumsum_iters_inner"][optimizer_config["name"]][0]["yvals"][-1]

        feas_rel = quad_prog.criteria["feas_rel_cumsum_iters_inner"][optimizer_config["name"]][0]["yvals"][-1]
        print(f"{config['problem_name']} ${optimizer_config['p']}$ & ${qp.num2tex(optimizer_config['tau'])}$ & ${qp.num2tex(optimizer_config['sigma'])}$ & ${qp.num2tex(subopt_rel)}$ & ${qp.num2tex(feas_rel)}$")



    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.grid(True)
    quad_prog.plot_criterion("subopt_rel_cumsum_iters_inner")

    plt.savefig(os.path.join('./results', f"{config['name']}_subopt_rel.pdf"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    #fig.suptitle(config["name"], fontsize=12)
    ax.grid(True)
    quad_prog.plot_criterion("feas_rel_cumsum_iters_inner")

    plt.savefig(os.path.join('./results', f"{config['name']}_feas_rel.pdf"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    #fig.suptitle(config["name"], fontsize=12)
    ax.grid(True)
    quad_prog.plot_criterion("r_pg_inf_cumsum_iters_inner")

    plt.savefig(os.path.join('./results', f"{config['name']}_r_pg_inf.pdf"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    #fig.suptitle(config["name"], fontsize=12)
    ax.grid(True)
    quad_prog.plot_criterion("r_comp_cumsum_iters_inner")

    plt.savefig(os.path.join('./results', f"{config['name']}_r_comp.pdf"))


