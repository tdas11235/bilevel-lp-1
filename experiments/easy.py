import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
from utils.bilevel import BilevelLPProblem
from utils.lp import ValueLP
from utils.restoration import RestorationNLP
from solver import BAPTRSolver


class LHSOnlyNonflatLP(BilevelLPProblem):
    def __init__(self, q_min, q_max):
        super().__init__(q_min, q_max, nx=2, m=2)

    def A_sym(self, q):
        return ca.vertcat(
            ca.horzcat(-(1 + q[0]), -1.0),
            ca.horzcat(1.0, 1 - q[0])
        )

    def b_sym(self, q):
        return ca.DM([-1.0, 2.0])
    

def c_smooth(nx, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, size=nx)

nx = 2
nq = 1
m = 2
c = np.ones(nx)
q_min =  np.zeros(nq)
q_max = 1.0 * np.ones(nq)

problem = LHSOnlyNonflatLP(q_min, q_max)
lp = ValueLP(c)
restoration = RestorationNLP(problem, rho=1.0)

solver = BAPTRSolver(
    problem, lp, restoration,
    max_iter=10
)

q0 = np.ones(nq) * 0.2
q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
if q is None:
    raise RuntimeError("Failed to find a feasible point!")

solutions = solver.solve(q)

print(solutions)