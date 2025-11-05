import unittest

from src.portfolio_qaoa import build_simple_quadratic_program, solve_with_qaoa


class PortfolioTests(unittest.TestCase):
    def test_build_qp(self):
        qp = build_simple_quadratic_program()
        # support both real QuadraticProgram and DummyQuadraticProgram
        if hasattr(qp, 'get_num_vars'):
            n = qp.get_num_vars()
        else:
            n = len(getattr(qp, 'variables', []))
        self.assertEqual(n, 4)

    def test_demo_solve_returns_result(self):
        qp = build_simple_quadratic_program()
        # call solve_with_qaoa in demo mode (returns dummy result if qiskit missing)
        res = solve_with_qaoa(qp, reps=1, seed=123, backend=None)
        # result should have attributes x (list-like) and fval (numeric)
        self.assertTrue(hasattr(res, 'x'))
        self.assertTrue(hasattr(res, 'fval'))
        self.assertEqual(len(res.x), 4)


if __name__ == '__main__':
    unittest.main()
