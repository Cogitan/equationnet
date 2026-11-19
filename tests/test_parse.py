import pytest
import sympy as sp

from equationnet import EquationError, parse_equation
from equationnet.codegen import explicit_function, residual_function


def test_ode_shorthand():
    eq = parse_equation("u_tt + c*u_t + k*u = 0")
    assert eq.indep == ["t"]
    assert eq.params == ["c", "k"]
    assert eq.derivs == {"u_t": ("t",), "u_tt": ("t", "t")}
    c, k, u, ut, utt = sp.symbols("c k u u_t u_tt")
    assert sp.simplify(eq.residual - (utt + c * ut + k * u)) == 0


def test_primes_and_caret():
    eq = parse_equation("u'' = -w^2*u")
    assert eq.derivs == {"u_tt": ("t", "t")}
    assert eq.params == ["w"]


def test_pde_variable_order_and_mixed_derivative():
    eq = parse_equation("u_t = D*u_xx + a*u_tx")
    assert eq.indep == ["x", "t"]
    assert "u_xt" in eq.derivs  # canonical order follows the variable order
    assert eq.derivs["u_xt"] == ("x", "t")


def test_known_values_are_substituted():
    eq = parse_equation("u_t = -k*u + b", known={"b": 2.0})
    assert eq.params == ["k"]
    assert sp.Symbol("b") not in eq.residual.free_symbols


def test_unknown_function():
    eq = parse_equation("u_tt + c*u_t + f(u) = 0")
    assert eq.params == ["c"]
    assert [u.name for u in eq.ufuncs] == ["f"]
    assert str(eq.ufuncs[0].arg) == "u"


def test_names_sympy_would_misread_are_parameters():
    # E, I, S, N, beta, gamma are sympy objects by default; here they must be plain constants
    eq = parse_equation("u_t = beta*S*u - gamma*u + E + I + N")
    assert set(eq.params) == {"beta", "S", "gamma", "E", "I", "N"}


@pytest.mark.parametrize("text, msg", [
    ("u_tt + u", "exactly one '='"),
    ("u_t = 1 = 2", "exactly one '='"),
    ("k*u = 0", "no derivative"),
    ("u_t = lambda*u", "reserved"),
    ("u_t = f(u, t)", "one argument"),
])
def test_clear_errors(text, msg):
    with pytest.raises(EquationError, match=msg):
        parse_equation(text)


def test_unknown_variable_letter():
    with pytest.raises(EquationError, match="not an independent variable"):
        parse_equation("u_tq = 0", indep=["t"])


def test_residual_code_runs_numpy_and_torch():
    import numpy as np
    import torch

    eq = parse_equation("u_t + u*u_x = nu*u_xx + sin(pi*x)")
    for backend, lib in (("numpy", np), ("torch", torch)):
        ns = {"np": np, "torch": torch}
        exec(residual_function(eq, backend), ns)
        V = {k: lib.ones(3) * 0.5 for k in ["x", "t", "u", "u_t", "u_x", "u_xx"]}
        P = {"nu": 0.1}
        out = ns["residual"](V, P, {})
        expected = 0.5 + 0.25 - 0.05 - np.sin(np.pi * 0.5)
        assert np.allclose(np.asarray(out), expected)


def test_explicit_needs_unique_solution():
    eq = parse_equation("u_t**2 = u")
    with pytest.raises(EquationError, match="uniquely"):
        explicit_function(eq, "rhs", "u_t", ["t", "u"])
