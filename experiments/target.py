import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import BAPTRSolver
from utils.restoration import RestorationNLP
from utils.lp import ValueLP
from utils.bilevel import BilevelLPProblem
import numpy as np
import casadi as ca


class TargetLP(BilevelLPProblem):
    def __init__(self):
        q_min = np.array([1.0, 1.0])
        q_max = np.array([5.0, 5.0])
        nx = 3
        m = 3
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([1.0, 4.0, 10.0])

    def A_sym(self, q):
        return ca.vertcat(
            ca.horzcat(-1.0, -1.0, -1.0),
            ca.horzcat(1.0, 1.0, 1.0),
            ca.horzcat(-1.0, -2.0, 0.0),
        )

    def b_sym(self, q):
        return ca.vertcat(
            -q[0]*q[1] - 5, q[0]*q[1] + 5, -q[0]**2
        )

    def cost(self):
        return self.c_vec


nx = 3
nq = 2
m = 3

problem = TargetLP()
c = problem.cost()
lp = ValueLP(c, lb_x=np.zeros(nx))
restoration = RestorationNLP(problem, rho=1.0)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=200
)

q0 = np.ones(nq) * 2
# q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
# if q is None:
#     raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q0)

print(solutions)
