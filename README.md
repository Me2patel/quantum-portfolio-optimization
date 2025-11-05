# Quantum Portfolio Optimization (QAOA) - Project

This project demonstrates a small, self-contained portfolio optimization example using Qiskit's QAOA and Qiskit Optimization.

## Structure
- `main.py` - Entry point. Builds a QuadraticProgram and solves it using QAOA + MinimumEigenOptimizer.
- `src/portfolio_qaoa.py` - Helper functions: build problem, run QAOA, evaluate result.
- `requirements.txt` - Python dependencies.
- `README.md` - This file.

## How to run
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the example:
   ```bash
   python main.py
   ```

## Notes
- This example is intentionally small (4 binary decision variables) so it runs quickly on a simulator.
- If you want to run on real hardware, configure an IBM Quantum account and replace the backend.
