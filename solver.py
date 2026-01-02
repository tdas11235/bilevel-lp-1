import numpy as np
from enum import Enum
from dataclasses import dataclass


class StepStatus(Enum):
    ACCEPTED = 0
    RESTART = 1
    TERMINATE = 2


@dataclass
class TerminatedPoint:
    q: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    fval: float


class BAPTRSolver:
    def __init__(
            self,
            problem,
            lp_solver,
            restoration_solver,
            *,
            eta=0.1,
            beta=0.5,
            delta0=1.0,
            tau=1e-6,
            eps=1e-6,
            kappa=0.1,
            eps_on=1e-6,
            eps_off=1e-8,
            max_iter=100,
            t_min=1e-8,
            term_tol=1e-8,
            slack_tol=1e-8,
            dir_tol=1e-10,
            damp=0.7,
            amp=1.5,
            switch_ratio=0.7,
            verbose=False
    ):
        """
        problem:            BilevelLPProblem
        lp_solver:          ValueLP
        restoration_solver: RestorationNLP
        """
        self.problem = problem
        self.lp = lp_solver
        self.restoration = restoration_solver
        # alogrithm params
        self.eta = eta
        self.beta = beta
        self.delta0 = delta0
        self.tau = tau
        self.eps = eps
        self.kappa = kappa
        self.eps_on = eps_on
        self.eps_off = eps_off
        self.max_iter = max_iter
        self.t_min = t_min
        self.term_tol = term_tol
        self.slack_tol = slack_tol
        self.dir_tol = dir_tol
        self.damp = damp
        self.amp = amp
        self.switch_ratio = switch_ratio
        self.verbose = verbose
        # internal global states
        self.terminated_points: list[TerminatedPoint] = []
        self.restore_count = 0
        self.tol_mode = False
        self.min_g = np.inf

    def _project_box(self, q):
        return np.clip(q, self.problem.q_min, self.problem.q_max)
    
    def _effective_gradient(self, q, g):
        g_eff = np.where(q <= self.problem.q_min + 1e-14, np.minimum(g, 0.0),
                         np.where(q >= self.problem.q_max - 1e-14, np.maximum(g, 0.0), g))
        return g_eff
    
    def _is_stationary(self, g_eff, q, fval):
        check1 = np.linalg.norm(g_eff) < self.eps
        if check1: return check1
        check2 = self.tol_mode and np.linalg.norm(g_eff) / max(1.0, np.abs(fval), np.linalg.norm(q)) < self.eps
        if check2: print("Reached stationary point within acceptable tolerance.")
        return check1 or check2
    
    def _already_terminated(self, q):
        for tp in self.terminated_points:
            if np.linalg.norm(q - tp.q, np.inf) < self.term_tol:
                return True
        return False
    
    def step(self):
        """
        Performs one trust region iteration with current q
        Returns StepStatus
        """
        q = self.q
        A = np.asarray(self.problem.eval_A(q))
        b = np.asarray(self.problem.eval_b(q)).flatten()
        # step-1: LP solve
        status, fval, x, lam = self.lp.solve(A, b)
        if fval is None: raise RuntimeError(f"LP solver failed to solve with {status}")
        self.last_x = x
        self.last_lambda = lam
        self.last_fval = fval
        # stpe-2: stationarity check
        g = self.problem.grad_L(q, x, lam)
        g_eff = self._effective_gradient(q, g)
        if self._is_stationary(g_eff, q, fval):
            if self._already_terminated(q):
                print("Global Termination reached.")
                return StepStatus.TERMINATE
            self.terminated_points.append(
                TerminatedPoint(
                    q=q.copy(),
                    x=x.copy(),
                    lam=lam.copy(),
                    fval=fval
                )
            )
            print("Stationary point reached.")
            return StepStatus.TERMINATE
        # step-3: descent direction
        norm_g = np.linalg.norm(g_eff)
        self.min_g = min(self.min_g, norm_g)
        d = -g_eff / norm_g
        # step-4: dual hysteresis for active set
        if self.active_prev is None:
            active = lam > self.eps_on
        else:
            active = self.active_prev.copy()
            active[lam > self.eps_on] = True
            active[lam < self.eps_off] = False
        # step-5: breakpoint prediction for active set
        phi0 = A @ x - b
        dphi = np.asarray(self.problem._dphi_fun(q, x, d)).flatten()
        t_break = np.inf
        for i in range(self.problem.m):
            if active[i]: continue
            if phi0[i] < -self.slack_tol and dphi[i] > self.dir_tol:
                t_i = -phi0[i] / dphi[i]
                if t_i > 0: t_break = min(t_break, t_i)
        # step-6: acceptance loop
        t = min(self.delta, t_break)
        accepted = False
        while t > self.t_min:
            dq = t * d
            q_trial = self._project_box(q + dq)
            dq_eff = q_trial - q
            # projection killed step
            if np.linalg.norm(dq_eff) == 0.0:
                if self._is_stationary(g_eff, q, fval):
                    if self._already_terminated(q):
                        print("Global Termination reached.")
                        return StepStatus.TERMINATE
                    self.terminated_points.append(
                        TerminatedPoint(
                            q=q.copy(),
                            x=x.copy(),
                            lam=lam.copy(),
                            fval=fval
                        )
                    )
                    print("Stationary point reached.")
                t *= self.beta
                continue
            A_t = np.asarray(self.problem.eval_A(q_trial))
            b_t = np.asarray(self.problem.eval_b(q_trial)).flatten()
            status_t, f_t, x_t, lam_t = self.lp.solve(A_t, b_t)
            # lp infeasible -> reject and shrink
            if f_t is None:
                print("Infeasible LP detected.")
                t *= self.beta
                continue
            delta_m = -np.dot(g_eff, dq_eff)
            if abs(delta_m) >= self.tau:
                rho = (fval - f_t) / delta_m
                if rho >= self.eta:
                    accepted = True
                    break
            else:
                if f_t < fval - self.tau:
                    accepted = True
                    break
            print(f"Not accepted due to failure in rho or tau test.")
            t *= self.beta
        # step-7: post acceptance
        if not accepted:
            print("No acceptible step found. Requesting restart.")
            return StepStatus.RESTART
        # step-8: Trust region update
        print(f"Step accepted with: {f_t}")
        if abs(delta_m) >= self.tau:
            if rho > 1.0 - self.kappa and np.array_equal(active, lam_t > self.eps_on):
                self.delta *= self.amp
            else:
                self.delta *= self.damp
        else:
            self.delta *= self.damp
        # accept step
        self.q = q_trial
        self.active_prev = active.copy()
        return StepStatus.ACCEPTED
        
    def solve(self, q0):
        self.q = self._project_box(np.asarray(q0, dtype=float))
        self.delta = self.delta0
        self.active_prev = None
        self.terminated_points: list[TerminatedPoint] = []
        for k in range(self.max_iter):
            print(f"\n--- ITER {k}, delta = {self.delta:.3e} ---")
            status = self.step()
            if status == StepStatus.ACCEPTED:
                continue
            if status == StepStatus.RESTART:
                print("Restart triggered.")
                self.restore_count += 1
                if (k >= self.switch_ratio * self.max_iter): self.tol_mode = True
                q_rest, _, _, stat = self.restoration.solve(
                    y_k=self.q, q_init=self.q
                )
                if q_rest is None: raise RuntimeError("Restoration failed.")
                self.q = self._project_box(q_rest)
                self.delta = self.delta0
                self.active_prev = None
                continue
            if status == StepStatus.TERMINATE:
                break
        solutions = [(tp.q, tp.x, tp.fval) for tp in self.terminated_points]
        # print(self.min_g)
        return solutions