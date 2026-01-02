import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
from utils.bilevel import BilevelLPProblem
from utils.lp import ValueLP
from utils.restoration import RestorationNLP
from solver import BAPTRSolver


class KnotLP(BilevelLPProblem):
    def __init__(self):
        q_min = np.ones(20)
        q_max = np.ones(20) * 5.0
        nx = 2
        m = 5
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([15.0, 5.0])

    def A_sym(self, q):
        return ca.vertcat(
            ca.horzcat(-1.0, 0.0),
            ca.horzcat(0.0, -1.0),
            ca.horzcat(-0.5, -0.8),
            ca.horzcat(-1.0, -1.0),
            ca.horzcat(1.0, 1.0),
        )

    def b_sym(self, q):
        row1 = ca.sum1(q[0:10]**2 - q[10:20])
        row2 = ca.sum1(q[10:20] * q[0:10] - 4)
        return ca.vertcat(
            -row1,
            -row2,
            -40.0,
            -100.0,
            100.0
        )

    def cost(self):
        return self.c_vec


nx = 2
nq = 20
m = 5

problem = KnotLP()
c = problem.cost()
lp = ValueLP(c, lb_x=np.array([20.0, 4.0]))
restoration = RestorationNLP(problem, rho=1.0)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=200
)

q0 = np.ones(nq) * (5.0)
q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
if q is None:
    raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q)

print(solutions)
