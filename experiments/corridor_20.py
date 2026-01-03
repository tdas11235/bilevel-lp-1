import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
import optuna as op
from utils.bilevel import BilevelLPProblem
from utils.lp import ValueLP
from utils.restoration import RestorationNLP
from solver import BAPTRSolver


class CorridorLP(BilevelLPProblem):
    def __init__(self):
        q_min = np.ones(20)
        q_max = np.ones(20) * 5.0
        nx = 2
        m = 4
        super().__init__(q_min, q_max, nx, m)
        self.c_vec = np.array([10.0, 2.0])

    def A_sym(self, q):
        return ca.vertcat(
            ca.horzcat(-1.0, 0.0),
            ca.horzcat(0.0, -1.0),
            ca.horzcat(-1.0, -1.0),
            ca.horzcat(1.0, 1.0),
        )

    def b_sym(self, q):
        row1 = ca.sumsqr(q[0:10] ** 2 - 2)
        row2 = ca.sum1(q[10:20] - 1)
        return ca.vertcat(
            -row1,
            -row2,
            -100.0,
            100.0
        )

    def cost(self):
        return self.c_vec


nx = 2
nq = 20
m = 4

problem = CorridorLP()
c = problem.cost()
lp = ValueLP(c, lb_x=np.array([2.0, 4.0]))
restoration = RestorationNLP(problem, rho=1.0)

def solveGrad():
    solver = BAPTRSolver(
        problem, lp, restoration,
        max_iter=200
    )

    q0 = np.ones(nq)
    # q, x, e, status = restoration.solve(y_k=q0, q_init=q0)
    # if q is None:
    #     raise RuntimeError("Failed to find a feasible point!")

    solutions = solver.solve(q0)

    print(solutions)


# solve using optuna (or BO)
def solOptuna(trial):
    q0 = []
    for i in range(nq):
        q0.append(trial.suggest_float(f'q_{i+1}', 1.0, 5.0))
    q0 = np.array(q0)
    A = np.asarray(problem.eval_A(q0))
    b = np.asarray(problem.eval_b(q0)).flatten()
    status, fval, x, lam = lp.solve(A, b)
    if fval is None: return 1e12
    return fval

def solveBO():
    study = op.create_study()
    study.optimize(solOptuna, n_trials=200, show_progress_bar=True)
    print(study.best_params.values())

solveBO()
# solveGrad()
