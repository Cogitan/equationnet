"""Print sympy expressions as plain Python for numpy or torch.

The generated scripts are the code that actually runs: equationnet renders a
script, executes it in-process, and hands the same text to the user.
"""

from __future__ import annotations

import sympy as sp
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.pycode import PythonCodePrinter

from .parse import Equation, EquationError


class _TorchPrinter(PythonCodePrinter):
    """Python printer that uses torch.* for math functions."""

    def __init__(self):
        super().__init__({"fully_qualified_modules": True, "standard": "python3"})

    def doprint(self, expr, assign_to=None):
        return super().doprint(expr, assign_to).replace("math.", "torch.")

    def _print_Abs(self, expr):
        return f"torch.abs({self._print(expr.args[0])})"


def numpy_code(expr: sp.Expr) -> str:
    return NumPyPrinter({"fully_qualified_modules": True}).doprint(expr).replace("numpy.", "np.")


def _numbers_as_floats(expr: sp.Expr) -> sp.Expr:
    """log(2), sqrt(2), exp(-1), pi: torch functions reject plain Python numbers, so every
    constant subexpression is evaluated to a float first."""
    return expr.replace(lambda e: getattr(e, "is_number", False) and not e.is_Number,
                        lambda e: sp.Float(e.evalf(17)))


def torch_code(expr: sp.Expr) -> str:
    return _TorchPrinter().doprint(_numbers_as_floats(expr))


# Names the generated code itself uses. A constant or column with one of these names
# keeps its name for the user (P["np"]) but gets a different name inside the script.
RESERVED = {"np", "torch", "math", "V", "P", "F", "residual", "sp", "scipy"}


def _safe(name: str) -> str:
    return f"{name}_eq" if name in RESERVED else name


def _safe_expr(expr: sp.Expr) -> sp.Expr:
    return expr.xreplace({s: sp.Symbol(_safe(str(s))) for s in expr.free_symbols if str(s) in RESERVED})


def _unpack_lines(names: list[str], source: str, indent: str = "    ") -> list[str]:
    return [f'{indent}{_safe(n)} = {source}["{n}"]' for n in names]


def residual_function(eq: Equation, backend: str) -> str:
    """def residual(V, P, F): V = values (variables, target, derivatives),
    P = fitted parameters, F = unknown functions (callables)."""
    code = torch_code if backend == "torch" else numpy_code
    names = [*eq.indep, eq.target, *eq.derivs]
    used = {str(s) for s in eq.residual.free_symbols}
    for u in eq.ufuncs:
        used |= {str(s) for s in u.arg.free_symbols}
    lines = ["def residual(V, P, F):",
             f'    """{eq.text}  ->  lhs - rhs (zero when the equation holds)."""']
    lines += _unpack_lines([n for n in names if n in used], "V")
    lines += _unpack_lines([p for p in eq.params if p in used], "P")
    for u in eq.ufuncs:
        lines.append(f'    {u.placeholder} = F["{u.name}"]({code(_safe_expr(u.arg))})')
    lines.append(f"    return {code(_safe_expr(eq.residual))}")
    return "\n".join(lines)


def solved_for(eq: Equation, name: str) -> sp.Expr:
    """Solve the residual for one derivative symbol (must be unique)."""
    sym = sp.Symbol(name)
    if not eq.residual.has(sym):
        raise EquationError(f"{name} does not appear in the equation")
    sols = sp.solve(eq.residual, sym, dict=False)
    if len(sols) != 1:
        raise EquationError(f"the equation cannot be solved uniquely for {name} "
                            f"({len(sols)} solutions)")
    return sols[0]


def explicit_function(eq: Equation, fname: str, solve_for: str, args: list[str]) -> str:
    """def <fname>(<args>, P, F): return <solve_for> written explicitly (numpy)."""
    expr = solved_for(eq, solve_for)
    used = {str(s) for s in expr.free_symbols}
    for u in eq.ufuncs:
        used |= {str(s) for s in u.arg.free_symbols}
    extra = sorted(used - set(args) - set(eq.params) - {u.placeholder for u in eq.ufuncs})
    if extra:
        raise EquationError(f"{solve_for} depends on {', '.join(extra)}, which this solver does not track")
    lines = [f"def {fname}({', '.join(_safe(a) for a in args)}, P, F):",
             f'    """{solve_for} from the equation {eq.text}"""']
    lines += _unpack_lines([p for p in eq.params if p in used], "P")
    for u in eq.ufuncs:
        lines.append(f'    {u.placeholder} = F["{u.name}"]({numpy_code(_safe_expr(u.arg))})')
    lines.append(f"    return {numpy_code(_safe_expr(expr))}")
    return "\n".join(lines)
