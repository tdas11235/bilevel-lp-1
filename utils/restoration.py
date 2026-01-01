import casadi as ca
import numpy as np


class RestorationNLP:
    def __init__(self, problem, rho, verbose=True):
        """
        problem:    BilevelProblem instance
        rho:        Regularization parameter
        """
        self.problem = problem
        self.rho = rho
        self.verbose = verbose
        self._build_solver()

    def _build_solver(self):
        nq = self.problem.nq
        nx = self.problem.nx
        m = self.problem.m
        # decision vars
        q = ca.MX.sym("q", nq)
        x = ca.MX.sym("x", nx)
        e = ca.MX.sym("e")
        # parameter (stability center)
        y = ca.MX.sym("y", nq)
        # expressions
        A_q = self.problem.A_sym(q)
        b_q = self.problem.b_sym(q)
        # constraints
        cons = []
        cons += [ca.mtimes(A_q, x) - b_q - e * ca.DM.ones(m)]
        # objective
        obj = e + (self.rho / 2.0) * ca.sumsqr(q - y)
        # bounds
        self.lbg = -np.inf * np.ones(m)
        self.ubg = np.zeros(m)
        self.lbx = np.concatenate([
            self.problem.q_min,
            np.zeros(nx),
            [-np.inf]
        ])
        self.ubx = np.concatenate([
            self.problem.q_max,
            np.inf * np.ones(nx),
            [0.0]
        ])
        # solver objects
        nlp = {
            "x": ca.vertcat(q, x, e),
            "f": obj,
            "g": ca.vertcat(*cons),
            "p": y
        }
        if self.verbose:
            opts = {
                "ipopt.print_level": 5,
                "print_time": True
            }
        else:
            opts = {
                "ipopt.print_level": 0,
                "print_time": False
            }
        self.solver = ca.nlpsol("restoration", "ipopt", nlp, opts)
        self.nx = nx
        self.nq = nq

    def solve(self, y_k, q_init=None, x_init=None, e_init=0.0):
        """
        Solve the restoration phase centered at y_k
        """
        if q_init is None:
            q_init = y_k
        if x_init is None:
            x_init = np.ones(self.nx)
        x0 = np.concatenate([q_init, x_init, [e_init]])
        # solve
        sol = self.solver(
            x0=x0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=y_k
        )
        stats = self.solver.stats()
        status = stats.get("return_status", "Unknown")
        print(status)
        if status not in ("Solve_Succeeded", "Converged_To_Acceptable_Point"):
            return None, None, None, status
        w = sol["x"].full().squeeze()
        q = w[:self.nq]
        x = w[self.nq: self.nq + self.nx]
        e = w[-1]
        return q, x, e, status