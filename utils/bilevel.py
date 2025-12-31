import numpy as np
import casadi as ca

class BilevelLPProblem:
    def __init__(self, q_min, q_max, nx, m):
        """
        q_min, q_max : box constraints on q
        nx           : dimension of x
        m            : number of constraints (rows of A)
        """
        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)
        self.nq = len(q_min)
        self.nx = nx
        self.m = m
        # build symbolic structure
        self._build_symbolic()
        self._build_grad_L()
        self._build_dphi_fun()
    
    def A_sym(self, q):
        """
        Symbolic A(q) of dims m-by-nx
        Must be overridden by user
        """
        raise NotImplementedError
    
    def b_sym(self, q):
        """
        Symbolic b(q) of dim m
        Must be overridden by user
        """
        raise NotImplementedError
    
    def _build_symbolic(self):
        q = ca.MX.sym("q", self.nq)
        A = self.A_sym(q)
        b = self.b_sym(q)
        self._A_fun = ca.Function("A_fun", [q], [A])
        self._b_fun = ca.Function("b_fun", [q], [b])
    
    def eval_A(self, q):
        """
        Return A(q)
        """
        return self._A_fun(q)
    
    def eval_b(self, q):
        """
        Return b(q)
        """
        return self._b_fun(q)
    
    def _build_grad_L(self):
        """
        Build casadi function for gradient of lagrangian
        """
        q = ca.MX.sym('q', self.nq)
        x = ca.MX.sym('x', self.nx)
        dual = ca.MX.sym('dual', self.m)
        A = self.A_sym(q)
        b = self.b_sym(q)
        expr = ca.dot(dual, ca.mtimes(A, x) - b)
        grad = ca.gradient(expr, q)
        self._grad_L_fun = ca.Function("grad_L", 
                                       [q, x, dual], [grad])
    
    def grad_L(self, q, x, dual):
        """
        Return gradient wrt q:
        dual.T @ (A(q) @ x - b(q))
        """
        g = self._grad_L_fun(q, x, dual)
        return np.asarray(g).flatten()
    
    def _build_dphi_fun(self):
        """
        Build casadi function for directional derivative of phi(q)
        """
        q = ca.MX.sym("q", self.nq)
        x = ca.MX.sym("x", self.nx)
        d = ca.MX.sym("d", self.nq)
        A = self.A_sym(q)
        b = self.b_sym(q)
        phi = ca.mtimes(A, x) - b
        dphi = ca.jacobian(phi, q) @ d
        self._dphi_fun = ca.Function("dphi", [q, x, d], [dphi])

