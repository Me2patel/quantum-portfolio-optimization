"""main.py

Entry point: builds a toy portfolio QuadraticProgram and solves it with QAOA + MinimumEigenOptimizer.
Designed to run on AerSimulator (local simulator).
"""

import sys
# Note: avoid importing qiskit at module import time so the script can run
# in demo mode when qiskit is not installed.

from src.portfolio_qaoa import build_simple_quadratic_program, solve_with_qaoa, evaluate_solution

def get_backend_simulator(seed=123):
    """
    Try to return an AerSimulator QuantumInstance in a robust way:
    - Prefer qiskit_aer.AerSimulator if available
    - Fall back to Aer provider if 'Aer' import is available in qiskit
    - Otherwise return None (demo mode)
    """
    # Option 1: qiskit-aer new package
    try:
        from qiskit_aer import AerSimulator
        backend = AerSimulator(seed_simulator=seed, seed_transpiler=seed)
        return backend
    except Exception:
        pass

    # Option 2: older qiskit Aer provider
    try:
        from qiskit import Aer
        backend = Aer.get_backend('aer_simulator')
        return backend
    except Exception:
        pass

    # No simulator available in this environment — return None and allow demo mode
    return None

def main():
    print("=== Quantum Portfolio Optimization (toy example) ===")
    qp = build_simple_quadratic_program()
    backend = get_backend_simulator()
    # Wrap backend into a QuantumInstance (used by QAOA in our helper) if available
    qi = None
    if backend is not None:
        try:
            from qiskit.utils import QuantumInstance
            qi = QuantumInstance(backend=backend, seed_simulator=123, seed_transpiler=123)
        except Exception:
            qi = None

    print("-- Solving with QAOA (reps=2) --")
    try:
        result = solve_with_qaoa(qp, reps=2, seed=123, backend=qi)
    except Exception as e:
        print("Error while solving with QAOA:", e)
        sys.exit(1)

    obj_val, selection = evaluate_solution(qp, result)
    print(f"Objective value: {obj_val}")
    print("Selected assets:")
    for k,v in selection.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
