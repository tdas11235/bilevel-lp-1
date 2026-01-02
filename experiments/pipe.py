import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
from utils.bilevel import BilevelLPProblem
from utils.lp import ValueLP
from utils.restoration import RestorationNLP
from solver import BAPTRSolver


class PipeLP(BilevelLPProblem):
    def __init__(self):
        q_min = np.array([1.0, 1.0])
        q_max = np.array([5.0, 5.0])
        nx = 2
        m = 4
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([10.0, 15.0])

    def A_sym(self, q):
        return ca.vertcat(
            ca.horzcat(-(q[0] + q[1]), -1.0),
            ca.horzcat(-1.0, -(q[0] * q[1])),
            ca.horzcat(-1.0, -1.0),
            ca.horzcat(1.0, 1.0)
        )

    def b_sym(self, q):
        return ca.vertcat(
            -50.0,
            -30.0,
            -20.0,
            20.0
        )

    def cost(self):
        return self.c_vec


nx = 2
nq = 2
m = 4

problem = PipeLP()
c = problem.cost()
lp = ValueLP(c, lb_x=np.ones(nx))
restoration = RestorationNLP(problem, rho=1.0)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=200
)

q0 = np.ones(nq)
q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
if q is None:
    raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q)

print(solutions)
