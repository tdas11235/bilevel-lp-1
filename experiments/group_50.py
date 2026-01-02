import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
from utils.bilevel import BilevelLPProblem
from utils.lp import ValueLP
from utils.restoration import RestorationNLP
from solver import BAPTRSolver


class GroupLP(BilevelLPProblem):
    def __init__(self):
        q_min = np.ones(50)
        q_max = np.ones(50) * 10.0
        nx = 100
        m = 52
        self._build_G()
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([float(i)+1.0 for i in range(100)])

    def _build_G(self):
        G_np = np.zeros((50, 50))
        for i in range(50):
            G_np[i][i] = 1.5
            if i>0: G_np[i][i-1] = -0.2
            if i<49: G_np[i][i+1] = -0.2
        self.G = ca.DM(G_np)

    def A_sym(self, q):
        rows_set1 = []
        for k in range(self.nq):
            row = np.zeros((1, self.nx))
            row[0, 2*k] = -1.0
            row[0, 2*k+1] = -1.0
            rows_set1.append(ca.DM(row))
        row_set2 = np.zeros((1, self.nx))
        row_set2[0, 0:60] = 1.0
        row_set3 = np.zeros((1, self.nx))
        row_set3[0, 39:100] = -1.0
        return ca.vertcat(*rows_set1, ca.DM(row_set2), ca.DM(row_set3))

    def b_sym(self, q):
        rhs1 = -ca.mtimes(self.G, q[0:50])
        q_slice2 = q[0:25]
        rhs2 = 300.0 + ca.sum1(1.5 * q_slice2 + 0.02 * q_slice2**2)
        q_slice3 = q[25:50]
        rhs3 = -150 - ca.sum1(2.0 * q_slice3 + 0.005 * q_slice3**2)
        return ca.vertcat(
            rhs1,
            rhs2,
            rhs3
        )

    def cost(self):
        return self.c_vec


nx = 100
nq = 50
m = 52

problem = GroupLP()
c = problem.cost()
lp = ValueLP(c, verbose=False)
restoration = RestorationNLP(problem, rho=1.0, verbose=False)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=2000, tau=1e-3, eps=1e-3,
    kappa=0.05,
    delta0=1e-2,
    amp=1.2,
    damp=0.8
)

q0 = np.ones(nq) * (2.0)
q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
if q is None:
    raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q)

print(solutions)
