import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import BAPTRSolver
from utils.restoration import RestorationNLP
from utils.lp import ValueLP
from utils.bilevel import BilevelLPProblem
import numpy as np
import casadi as ca


class WedgeLP(BilevelLPProblem):
    def __init__(self):
        q_min = np.array([-np.inf, -np.inf])
        q_max = np.array([np.inf, np.inf])
        nx = 3
        m = 2
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([2.0, 3.0, 1.0])

    def A_sym(self, q):
        return ca.vertcat(
            ca.horzcat(-(q[0] * q[1]), -(q[0] ** 3), -1.0),
            ca.horzcat(-(q[1] ** 2), -(q[0] + q[1]), -2)
        )

    def b_sym(self, q):
        return ca.DM([-20.0, -15.0])

    def cost(self):
        return self.c_vec


nx = 3
nq = 2
m = 2

problem = WedgeLP()
c = problem.cost()
lp = ValueLP(c, lb_x=np.ones(nx))
restoration = RestorationNLP(problem, rho=1.0)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=100
)

q0 = np.ones(nq) * (-2)
# q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
# if q is None:
#     raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q0)

print(solutions)
