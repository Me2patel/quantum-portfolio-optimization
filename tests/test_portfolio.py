def test_build_qp():
    from src.portfolio_qaoa import build_simple_quadratic_program
    qp = build_simple_quadratic_program()
    assert qp.get_num_vars() == 4
