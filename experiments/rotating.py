import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
from utils.bilevel import BilevelLPProblem
from utils.lp import ValueLP
from utils.restoration import RestorationNLP
from solver import BAPTRSolver


class LP2(BilevelLPProblem):
    def __init__(self):
        q_min = np.array([-5.0])
        q_max = np.array([5.0])
        nx = 2
        m = 1
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([1.0, 1.0])

    def A_sym(self, q):
        return ca.MX(ca.horzcat(-q[0], -1.0))

    def b_sym(self, q):
        return ca.MX([-10.0])

    def cost(self):
        return self.c_vec

    

nx = 2
nq = 1
m = 2
q_min = np.ones(nq) * 2.0
q_max = 5.0 * np.ones(nq)

problem = LP2()
c = problem.cost()
lp = ValueLP(c)
restoration = RestorationNLP(problem, rho=1.0)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=10
)

q0 = np.zeros(nq)
# q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
# if q is None:
#     raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q0)

print(solutions)
