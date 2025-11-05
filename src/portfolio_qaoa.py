"""src/portfolio_qaoa.py

Utilities for building a simple QuadraticProgram for a toy portfolio optimization
and solving it with QAOA via MinimumEigenOptimizer.
"""

from typing import Dict, Tuple, List
try:
    import numpy as np
except Exception:
    np = None

# Qiskit imports (import inside functions to allow graceful error messages)
def build_simple_quadratic_program():
    """Builds a toy QuadraticProgram with 4 binary variables.
    The objective is a quadratic form representing (negative) expected return
    and pairwise interactions (e.g., covariance / penalty terms).
    Returns the QuadraticProgram object.
    """
    try:
        from qiskit_optimization import QuadraticProgram
    except Exception:
        # Lightweight fallback used for environments without qiskit-optimization
        class DummyVariable:
            def __init__(self, name: str):
                self.name = name

        class QuadraticProgram:
            def __init__(self):
                self.variables: List[DummyVariable] = []
                self._linear = {}
                self._quadratic = {}

            def binary_var(self, name: str):
                v = DummyVariable(name)
                self.variables.append(v)
                return v

            def get_num_vars(self) -> int:
                return len(self.variables)

            def minimize(self, linear=None, quadratic=None, constant=0.0):
                # store objective; no solving performed in dummy
                self._linear = dict(linear) if linear is not None else {}
                self._quadratic = dict(quadratic) if quadratic is not None else {}

    qp = QuadraticProgram()
    # Create 4 binary decision variables x0..x3 (include 4 assets)
    for i in range(4):
        qp.binary_var(name=f"x{i}")

    # Example: maximize expected return with penalty for selecting too many assets.
    # QuadraticProgram in Qiskit defaults to minimization, so we convert maximize to minimize by negation.
    # We'll build a simple quadratic objective: -sum(mu_i x_i) + lam * (sum x_i - k)^2 as penalty.
    # example expected returns (use numpy if available, otherwise plain list)
    mu = np.array([0.10, 0.12, 0.07, 0.09]) if np is not None else [0.10, 0.12, 0.07, 0.09]
    lam = 0.5   # penalty strength
    k = 2       # target number of assets to pick

    # Linear terms: -mu_i (use variable name strings directly)
    linear = { f"x{i}": float(-mu[i]) for i in range(4) }

    # Quadratic penalty for (sum x_i - k)^2 = sum_i x_i^2 + 2 sum_{i<j} x_i x_j - 2k sum_i x_i + k^2
    # Since x_i^2 = x_i for binary variables, include them in linear term
    for i in range(4):
        linear[f"x{i}"] = linear.get(f"x{i}", 0.0) + float(lam * 1.0)  # from x_i^2 term

    quadratic = {}
    # pairwise terms 2*lam for i<j
    for i in range(4):
        for j in range(i+1, 4):
            quadratic[(f"x{i}", f"x{j}")] = float(2.0 * lam)

    # adjust linear terms for -2k*lam * x_i
    for i in range(4):
        linear[f"x{i}"] = linear[f"x{i}"] + float(-2.0 * lam * k)

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp

def solve_with_qaoa(qp, reps=2, seed=None, optimizer=None, backend=None):
    """Solve given QuadraticProgram using QAOA wrapped in MinimumEigenOptimizer.

    Returns the optimization result object from the optimizer.
    """
    try:
        from qiskit.algorithms import QAOA
        from qiskit.algorithms.optimizers import COBYLA, SLSQP
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
    except Exception:
        # Qiskit not available in this environment — return a dummy result so
        # callers (and main demo) can run in demo mode without heavy deps.
        n = qp.get_num_vars() if hasattr(qp, 'get_num_vars') else len(getattr(qp, 'variables', []))
        # simple heuristic: pick first 2 assets (or fewer if n<2)
        sel = [1 if i < min(2, n) else 0 for i in range(n)]

        # try to compute objective value from stored linear/quadratic if present
        fval = 0.0
        linear = getattr(qp, '_linear', {}) or {}
        quadratic = getattr(qp, '_quadratic', {}) or {}
        for i in range(n):
            name = f"x{i}"
            fval += float(linear.get(name, 0.0)) * sel[i]
        for (a, b), val in quadratic.items():
            # keys are expected as ("x0","x1") strings
            try:
                ia = int(a[1:]) if isinstance(a, str) and a.startswith('x') else None
                ib = int(b[1:]) if isinstance(b, str) and b.startswith('x') else None
                if ia is not None and ib is not None:
                    fval += float(val) * sel[ia] * sel[ib]
            except Exception:
                # ignore malformed keys
                pass

        class DummyResult:
            def __init__(self, x, fval):
                self.x = x
                self.fval = fval

        return DummyResult(sel, fval)

    # Set up a classical optimizer if none provided
    if optimizer is None:
        optimizer = COBYLA(maxiter=200)

    # Create QAOA instance
    qaoa = QAOA(optimizer=optimizer, reps=reps, quantum_instance=backend)

    # Wrap with MinimumEigenOptimizer
    meo = MinimumEigenOptimizer(min_eigen_solver=qaoa)
    result = meo.solve(qp)
    return result

def evaluate_solution(qp, result) -> Tuple[float, Dict[str,int]]:
    """Evaluate objective value and return selected assets as dict."""
    x = {var.name: int(val) for var, val in zip(qp.variables, result.x)}
    obj_val = result.fval
    return obj_val, x
