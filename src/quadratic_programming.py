import numpy as np

import compot
import compot.misc.benchmarks as benchmarks
import compot.calculus.function as fun

import anisotropic.function as prox
import anisotropic.alm as anisotropic_alm
import maros_meszaros as mm

import scipy.sparse as sp

import qpalm



def convert_maros_meszaros_to_equalities(prob):
    Q = prob.P
    c = prob.q
    A = prob.A
    b = prob.b
    lb = prob.lb
    ub = prob.ub
    G = prob.G
    h = prob.h

    # Flatten vectors
    c = c.ravel()
    b = b.ravel() if b is not None else np.zeros(0)
    lb = lb.ravel()
    ub = ub.ravel()
    h = h.ravel() if h is not None else np.zeros(0)

    if G is None or G.shape[1] == 0:
        A_new = A
        b_new = b
        Q_new = Q
        c_new = c
        lb_new = lb
        ub_new = ub
    else:
        m_ineq = G.shape[0]
        I_slack = sp.eye(m_ineq, format="csr")
        A_ineq = sp.hstack([G, I_slack])

        if A is not None:
            if A.shape[1] < G.shape[1] + m_ineq:
                pad = sp.csr_matrix((A.shape[0], G.shape[1] + m_ineq - A.shape[1]))
                A_padded = sp.hstack([A, pad])
                A_new = sp.vstack([A_padded, A_ineq])
            else:
                A_new = sp.vstack([A, A_ineq])
            b_new = np.concatenate([b, h])
        else:
            A_new = A_ineq
            b_new = h

        Q_new = sp.block_diag((Q, sp.csr_matrix((m_ineq, m_ineq))), format="csr")
        c_new = np.concatenate([c, np.zeros(m_ineq)])
        lb_new = np.concatenate([lb, np.zeros(m_ineq)])
        ub_new = np.concatenate([ub, np.full(m_ineq, np.inf)])

    return {
        'Q': Q_new,
        'c': c_new,
        'A': A_new,
        'b': b_new,
        'l': lb_new,
        'u': ub_new
    }

def find_maros_meszaros_problem(name):
    maros_meszaros = mm.MarosMeszaros()
    qp = None
    for prob in maros_meszaros:
        if prob.name == name:
            qp = convert_maros_meszaros_to_equalities(prob)
            break
    if qp is None:
        raise ValueError(f"Problem {name} not found")

    return qp

def solve_standard_form_qp_qpalm(parsed_qp):
    """
    Solve 0.5 x^T Q x + c^T x + c0
    s.t. A x = b   and   l <= x <= u
    using QPALM (bmin <= A*x <= bmax formulation).

    parsed_qp keys: "Q","c","A","b","l","u","c0"
    """

    # --- unpack ---
    Q = parsed_qp["Q"]
    c = np.asarray(parsed_qp["c"], dtype=float)
    A = parsed_qp["A"]
    b = np.asarray(parsed_qp["b"], dtype=float).ravel()
    l_bounds = np.asarray(parsed_qp["l"].copy(), dtype=float).ravel()
    u_bounds = np.asarray(parsed_qp["u"].copy(), dtype=float).ravel()
    c0 = float(parsed_qp.get("c0", 0.0))

    n = int(Q.shape[0])
    # ensure Q is CSC (QPALM expects sparse CSC)
    Q_csc = sp.csc_matrix(Q)

    # ensure A is a (m_eq x n) sparse matrix (may be empty)
    if A is None:
        A_eq = sp.csc_matrix((0, n))
        m_eq = 0
    else:
        A_eq = sp.csc_matrix(A)
        m_eq = int(A_eq.shape[0])

    # equality rows: represent Ax = b as bmin = bmax = b
    bmin_eq = b.copy()
    bmax_eq = b.copy()

    # variable bounds: only add rows for variables where at least one bound is finite
    mask = np.isfinite(l_bounds) | np.isfinite(u_bounds)
    idx = np.nonzero(mask)[0]
    k = idx.size

    if k > 0:
        # build a small "identity" block with one row per bounded variable
        rows = np.arange(k, dtype=int)
        cols = idx.astype(int)
        data = np.ones(k, dtype=float)
        I_bounds = sp.csc_matrix((data, (rows, cols)), shape=(k, n))
        # stack eq constraints and identity-rows for bounds
        A_aug = sp.vstack([A_eq, I_bounds], format="csc")
        # corresponding bmin/bmax for the appended rows:
        bmin_bounds = l_bounds[idx]
        bmax_bounds = u_bounds[idx]
        bmin = np.concatenate([bmin_eq, bmin_bounds])
        bmax = np.concatenate([bmax_eq, bmax_bounds])
    else:
        A_aug = A_eq
        bmin = bmin_eq
        bmax = bmax_eq

    m_tot = int(A_aug.shape[0])

    # --- fill QPALM data ---
    data = qpalm.Data(n, m_tot)
    data.Q = Q_csc
    data.q = c
    data.A = A_aug
    data.bmin = bmin
    data.bmax = bmax

    # --- settings (tweak tolerances if you want) ---
    settings = qpalm.Settings()
    settings.eps_abs = 1e-13
    settings.eps_rel = 1e-13
    settings.verbose = False

    # --- solve ---
    solver = qpalm.Solver(data, settings)
    solver.solve()

    # --- read result ---
    x = solver.solution.x              # primal
    info = solver.info                 # QPALMInfo struct (status, objective, residuals, iters...). :contentReference[oaicite:1]{index=1}

    # objective: prefer solver.info.objective if available, otherwise recompute
    obj_from_info = getattr(info, "objective", np.nan)
    if np.isfinite(obj_from_info):
        objective_value = float(obj_from_info) + c0
    else:
        # fall back to explicit computation
        objective_value = float(0.5 * x @ (Q_csc.dot(x)) + c @ x + c0)

    # Build a helpful status/message
    status_str = getattr(info, "status", None)
    if status_str is None:
        # some wrappers expose status_val instead
        status_val = getattr(info, "status_val", None)
        status_str = f"status_val={status_val}"

    # If you need to map duals back:
    # solver.solution.y contains multipliers for the augmented constraint vector:
    #   first `m_eq` entries -> multipliers for A x = b
    #   next `k` entries  -> multipliers for variable bounds (in order idx)
    return {
        "x": x,
        "objective_value": objective_value,
        "status": status_str,
        "info": info,              # contains iter, pri_res_norm, dua_res_norm, objective, etc.
        "message": "Solved with QPALM (bmin <= A x <= bmax)."
    }

class QuadraticProgram(anisotropic_alm.SaddlePointProblem):
    #x_init, y_init, A, b, proxable_primal, diffable_primal, proxable_dual):
    def __init__(self, x_init, y_init, Q, c, c0, A, b, lb, ub):
        self.Q = Q
        self.c = c
        self.c0 = c0
        #print("min eigenvalue", scipy.sparse.linalg.eigsh(Q, k=1, which='SA', maxiter=100000)[0].min())
        super().__init__(x_init, y_init,
            fun.SparseMatrixLinearTransform(A),
            b,
            fun.IndicatorBox(l=lb, u=ub),
            fun.QuadraticFunction(fun.SparseMatrixLinearTransform(Q), c, c0),
            prox.Constant(C=0)
        )

    def eval_dual(self, y):
         return -np.inf

    def eval_primal(self, x):
         return 0.5 * (x.T @ self.Q @ x) + (self.c.T @ x) + self.c0


class QuadraticProgramming(benchmarks.Benchmark):
    def setup(self):
        qp = find_maros_meszaros_problem(self.config["problem_name"])
        m, n = qp["A"].shape

        problem_data = {}
        problem_data["qp"] = qp

        self.x_init = np.minimum(qp["u"], np.maximum(qp["l"], np.random.randn(n)))
        self.y_init = np.zeros(m)
        result = solve_standard_form_qp_qpalm(qp)

        self.opt = result["objective_value"]#get_opt_value("data.json", self.config["problem_name"])

        return problem_data

    def get_filename(self):
        return ("results/" + self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"]))

    def setup_problem(self):
        return QuadraticProgram(self.x_init, self.y_init, self.problem_data["qp"]["Q"], self.problem_data["qp"]["c"], 0.0, self.problem_data["qp"]["A"], self.problem_data["qp"]["b"], self.problem_data["qp"]["l"], self.problem_data["qp"]["u"])



    def setup_optimizer(self, optimizer_config, problem, callback):
        params = anisotropic_alm.Parameters()
        params.class_oracle = optimizer_config["class_oracle"]
        params.params_oracle = optimizer_config["params_oracle"]
        params.maxit = optimizer_config["maxit"]
        params.tau = optimizer_config["tau"]
        params.sigma = optimizer_config["sigma"]
        params.bounds = optimizer_config["bounds"]
        params.proxfun_primal = optimizer_config["proxfun_primal"]
        params.proxfun_dual = optimizer_config["proxfun_dual"]
        params.maxit_cumsum_inner = optimizer_config["max_cumsum_iters_inner"]
        params.tolerance = optimizer_config["tolerance"]

        optimizer = optimizer_config["class"](params, problem, callback)

        return optimizer

    def get_criterion_keys(self):
        return ["subopt_cumsum_iters_inner", "subopt_rel_cumsum_iters_inner", "feas_cumsum_iters_inner", "feas_rel_cumsum_iters_inner", "r_pg_inf_cumsum_iters_inner", "r_comp_cumsum_iters_inner"]

    def eval_criteria(self, problem, variable, status):
        x, s, y = variable
        primal_value = problem.eval_primal(x)

        subopt = np.abs(primal_value - self.opt)
        feas = np.linalg.norm((self.problem_data["qp"]["A"] @ x) - self.problem_data["qp"]["b"], np.inf)

        feas_rel = feas / (1 + np.linalg.norm(self.problem_data["qp"]["b"], np.inf))
        subopt_rel = subopt / (1 + abs(self.opt))

        # gradient w.r.t. x (without bound multipliers z)
        # handle Q as array or sparse
        if hasattr(problem.Q, "dot"):
            Qx = problem.Q.dot(x)
        else:
            Qx = problem.Q @ x
        g = Qx + problem.c + problem.A.apply_transpose(y)

        # projected gradient residual:
        # projection onto [l,u] with broadcasting for infinite bounds
        x_minus_g = x - g
        proj = problem.proxable_primal.eval_prox(x_minus_g, 1.) # P_[l,u](x - g)
        delta = proj - x
        r_pg_inf = np.max(np.abs(delta))
        #r_pg_2 = np.linalg.norm(delta, 2)

        # complementarity proxy (optional)
        z_l = np.maximum(-g, 0.0)  # candidate lower multipliers
        z_u = np.maximum(g, 0.0)  # candidate upper multipliers
        # Lower bound terms
        has_lower = np.isfinite(problem.proxable_primal._l)
        r_low = np.zeros_like(x)
        if np.any(has_lower):
            r_low[has_lower] = (x[has_lower] - problem.proxable_primal._l[has_lower]) * z_l[has_lower]

        # Upper bound terms
        has_upper = np.isfinite(problem.proxable_primal._u)
        r_up = np.zeros_like(x)
        if np.any(has_upper):
            r_up[has_upper] = (problem.proxable_primal._u[has_upper] - x[has_upper]) * z_u[has_upper]

        # Stack and compute norms
        r_all = np.concatenate([r_low[has_lower], r_up[has_upper]])
        if r_all.size == 0:
            r_comp = 0
        else:
            r_comp = np.linalg.norm(r_all, ord=np.inf)

        criteria = {
            "subopt_cumsum_iters_inner": {
                "x": status.cumsum_iters_inner,
                "y": subopt
            },
            "subopt_rel_cumsum_iters_inner": {
                "x": status.cumsum_iters_inner,
                "y": subopt_rel
            },
            "feas_cumsum_iters_inner": {
                "x": status.cumsum_iters_inner,
                "y": feas
            },
            "feas_rel_cumsum_iters_inner": {
                "x": status.cumsum_iters_inner,
                "y": feas_rel
            },
            "r_pg_inf_cumsum_iters_inner": {
                "x": status.cumsum_iters_inner,
                "y": r_pg_inf
            },
            "r_comp_cumsum_iters_inner": {
                "x": status.cumsum_iters_inner,
                "y": r_comp
            }
        }

        return criteria

    def get_refvals(self):
        return {"subopt_cumsum_iters_inner": 0, "subopt_rel_cumsum_iters_inner": 0, "feas_cumsum_iters_inner": 0, "feas_rel_cumsum_iters_inner": 0, "r_pg_inf_cumsum_iters_inner": 0, "r_comp_cumsum_iters_inner": 0}

    def get_axis_labels(self):
        return None


markers = ["^", "x", "p", "d"]
linestyles = ["solid", "dashdot", "dashed", "dashdot"]

def num2tex(x, d=2):
    if x == 1:
        return "1"
    man = round(benchmarks.fman10(x), d)
    exp = benchmarks.fexp10(x)
    if man != 1:
        str = f"{man}"
        if exp != 0:
            str += " \\cdot "
    else:
        str = ""

    if exp == 1:
        str += "10"
    elif exp != 0:
        str += f"10^{{{exp}}}"

    return  str

def expand_optimizer_configs(configs, maxit, max_cumsum_iters_inner, subopt_rel, feas_rel, class_oracle, params_oracle, eps, linewidth):
    configs_expanded = []
    for i, config in enumerate(configs):
        config_expanded = {
            "marker": markers[i],
            "color": "black",
            "linestyle": linestyles[i],
            "name": f"p{config['p']}_tau{config['tau']}_sigma{config['sigma']}",
            "class": anisotropic_alm.AugmentedLagrangianMethod,
            "label": f"$p={config['p']},\\sigma={num2tex(config['sigma'])},\\tau={num2tex(config['tau'])}$",
            "maxit": maxit,
            "max_cumsum_iters_inner": max_cumsum_iters_inner,
            "subopt_rel": subopt_rel,
            "feas_rel": feas_rel,
            "class_oracle": class_oracle,
            "params_oracle": params_oracle,
            "proxfun_primal": prox.IsotropicProxFunction(
                prox.PowerElemProxFunction(config['p'])
            ),
            "proxfun_dual": prox.IsotropicProxFunction(
                prox.PowerElemProxFunction(config['p'])
            ),
            "tolerance": lambda k: eps / pow((1 + k), config['p']),
            "sigma": config['sigma'],
            "tau": config['tau'],
            "p": config['p'],
            "bounds": np.inf,
            "markevery": 1,
            "plotevery": 1,
            "linewidth": linewidth
        }
        configs_expanded.append(config_expanded)

    return configs_expanded
