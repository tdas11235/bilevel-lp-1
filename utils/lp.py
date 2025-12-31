import numpy as np
import highspy as hp


class ValueLP:
    def __init__(self, c, verbose=True):
        self.c = np.asarray(c)
        self.verbose = verbose

    def solve(self, A, b):
        """
        Solve:
        min c.T x
        s.t. A x <= b, x >= 0
        """
        m, n = A.shape
        highs = hp.Highs()
        if not self.verbose: highs.silent()
        # variables
        lb_x = np.zeros(n)
        ub_x = np.full(n, np.inf)
        highs.addVars(n, lb_x, ub_x)
        # objective
        idx = np.arange(n, dtype=np.int32)
        cost = np.asarray(self.c, dtype=np.float64)
        highs.changeColsCost(n, idx, cost)
        # constraints
        for i in range(m):
            idx = np.arange(n, dtype=np.int32)
            val = np.asarray(A[i, :], dtype=np.float64)
            highs.addRow(-np.inf, b[i], n, idx, val)
        # solve
        highs.run()
        status = highs.getModelStatus()
        if status != hp.HighsModelStatus.kOptimal:
            return status, None, None, None
        # primal, dual variables
        x = np.array(highs.getSolution().col_value)
        lam = np.array(highs.getSolution().row_dual)
        fval = highs.getObjectiveValue()
        return status, fval, x, lam
