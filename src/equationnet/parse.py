"""Turn an equation typed as text into a residual expression.

Notation (all optional except the equation itself):
  u_t, u_xx, u_xt   partial derivatives of the target u (one letter per variable)
  u', u''           derivatives for a single independent variable
  ^ or **           powers
  f(u)              an unknown function of one argument, learned from data
  sin, cos, exp ... ordinary math functions

Every other name is a parameter. Parameters without a known value are fitted.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

MATH_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs,
    # other spellings people use for the same functions: read as what they mean, never
    # as an unknown function to learn
    "ln": sp.log, "log10": lambda z: sp.log(z, 10), "log2": lambda z: sp.log(z, 2),
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "arcsinh": sp.asinh, "arccosh": sp.acosh, "arctanh": sp.atanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "sec": sp.sec, "csc": sp.csc, "cot": sp.cot, "erf": sp.erf, "erfc": sp.erfc,
}
CONSTANTS = {"pi": sp.pi}
# Math names this version does not support (not smooth, or more than one argument).
# Written as calls they would otherwise become unknown functions to learn, silently.
UNSUPPORTED_FUNCS = {"sign", "sgn", "max", "min", "floor", "ceil", "round", "heaviside",
                     "Heaviside", "step", "mod", "pow", "atan2", "hypot", "gamma", "factorial"}


class EquationError(ValueError):
    """The equation text cannot be read. The message says what to change."""


@dataclass
class UnknownFunction:
    placeholder: str  # symbol name standing in for this application, e.g. "f_0"
    name: str         # the function's name, e.g. "f"
    arg: sp.Expr      # its argument, e.g. u


@dataclass
class Equation:
    text: str
    target: str
    indep: list[str]
    residual: sp.Expr                  # lhs - rhs, unknown functions replaced by placeholders
    derivs: dict[str, tuple[str, ...]]  # "u_tt" -> ("t", "t")
    params: list[str]                  # parameters still to fit
    known: dict[str, float] = field(default_factory=dict)
    ufuncs: list[UnknownFunction] = field(default_factory=list)

    @property
    def ufunc_names(self) -> list[str]:
        return sorted({u.name for u in self.ufuncs})

    @property
    def time(self) -> str:
        """The time variable: 't' if there is one; otherwise the only variable whose
        derivatives are all first order while every other variable has a higher one
        (u_s = D*u_xx: s); otherwise the last variable."""
        if "t" in self.indep:
            return "t"
        if len(self.indep) == 1:
            return self.indep[0]
        first = [v for v in self.indep if self.order_in(v) == 1]
        if len(first) == 1 and all(self.order_in(v) != 1 for v in self.indep if v != first[0]):
            return first[0]
        return self.indep[-1]

    def order_in(self, var: str) -> int:
        return max((seq.count(var) for seq in self.derivs.values()), default=0)

    def describe(self) -> str:
        parts = [f"target {self.target}({', '.join(self.indep)})"]
        if self.params:
            parts.append("fit " + ", ".join(self.params))
        if self.known:
            parts.append("known " + ", ".join(f"{k}={v:g}" for k, v in self.known.items()))
        if self.ufuncs:
            parts.append("learn " + ", ".join(f"{u.name}({u.arg})" for u in self.ufuncs))
        return "; ".join(parts)


def _deriv_name(target: str, seq: tuple[str, ...]) -> str:
    return f"{target}_{''.join(seq)}"


MAX_LENGTH = 2000          # characters in an equation
MAX_EXPONENT = 1000        # largest numeric exponent allowed (9**9**9 would never finish)
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \t+-*/^(),='.")
_LOOKALIKES = {"−": "-", "–": "-", "—": "-", "×": "*", "·": "*",
               "⋅": "*", "’": "'", "′": "'", "″": "''", "²": "^2"}


def check_text(text: str) -> None:
    """Allow only what an equation needs: letters, digits, _, spaces, + - * / ^ ( ) , = '
    and a decimal point inside a number. Anything else (quotes, brackets, dots after names,
    #, ;, newlines, backslashes, unicode look-alikes) stops here, before any parsing, with
    the character named. This is what keeps equation text from running code."""
    if not isinstance(text, str):
        raise EquationError(f"the equation must be text, not {type(text).__name__}")
    if len(text) > MAX_LENGTH:
        raise EquationError(f"the equation is {len(text)} characters long; the limit is {MAX_LENGTH}")
    for i, ch in enumerate(text):
        if ch not in _ALLOWED:
            hint = f"; type '{_LOOKALIKES[ch]}' instead" if ch in _LOOKALIKES else ""
            shown = repr(ch) if ch.isprintable() else f"a {'newline' if ch in chr(10) + chr(13) else 'control'} character"
            raise EquationError(f"the equation contains {shown} (U+{ord(ch):04X}) at position {i + 1}, "
                                f"which cannot be used in an equation{hint}")
    for m in re.finditer(r"\.", text):
        before = text[m.start() - 1] if m.start() else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if not (before.isdigit() or after.isdigit()) or (before.isalpha() or before == "_"):
            raise EquationError(f"'.' at position {m.start() + 1} is only allowed inside a number (e.g. 0.5)")
    for m in re.finditer(r"[A-Za-z_]\w*", text):
        if m.group(0).startswith("_") or "__" in m.group(0):
            raise EquationError(f"'{m.group(0)}': names cannot start with '_' or contain '__'")


def _syntax_hint(text: str, exc: Exception) -> str:
    """Say what is probably wrong in words, instead of Python's 'invalid syntax'."""
    m = re.search(r"([A-Za-z0-9_')])\s+([A-Za-z0-9_(])", text)
    if m and not re.search(r"[-+*/^=,(]\s*$", text[:m.start() + 1]):
        tail = (" (a function call has no space before its bracket: f(u))" if m.group(2) == "(" else "")
        return (f"is a '*' missing between '{m.group(1)}' and '{m.group(2)}' at position "
                f"{m.start() + 2}? Write multiplication with '*', e.g. k*u{tail}")
    m = re.search(r"\d([A-Za-z_(])", text)
    if m and not re.match(r"[eE][-+]?\d", text[m.start(1):]):
        return (f"is a '*' missing after the number at position {m.start() + 1}? "
                "Write e.g. 2*u, not 2u")
    for side in text.split("="):
        s = side.strip()
        if not s:
            return "one side of '=' is empty"
        if s[-1] in "+-*/^,":
            return f"'{s}' ends with '{s[-1]}' and nothing after it"
    if text.count("(") != text.count(")"):
        return f"the brackets do not match ({text.count('(')} '(' and {text.count(')')} ')')"
    msg = str(exc)
    if "invalid syntax" in msg or "<unknown>" in msg:
        return "check for a missing '*', an unclosed bracket, or an operator with nothing after it"
    return msg


def check_name(name: str, what: str) -> None:
    """Column and target names become names in the generated script: a letter, then
    letters, digits or _."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise EquationError(f"the {what} name {name!r} cannot be used: a name must start with a letter "
                            "and contain only letters, digits and _ (rename it)")
    if keyword.iskeyword(name):
        raise EquationError(f"the {what} name '{name}' is a reserved Python word; rename it")
    if name in MATH_FUNCS or name in CONSTANTS:
        raise EquationError(f"the {what} name '{name}' is also a math name in equations; rename it")


def _check_powers(expr) -> None:
    """Refuse numeric exponents above MAX_EXPONENT before sympy tries to compute them."""
    def value(e):
        if e.is_Number:
            return float(e)
        if e.is_Pow:
            b, x = value(e.base), value(e.exp)
            if abs(x) > MAX_EXPONENT:
                raise EquationError(f"the exponent {x:.4g} is too large (limit {MAX_EXPONENT})")
            try:
                return float(b) ** x
            except (OverflowError, ZeroDivisionError):
                raise EquationError("a number in the equation is too large to compute") from None
        args = [value(a) for a in e.args]
        if e.is_Add:
            return sum(args)
        if e.is_Mul:
            out = 1.0
            for a in args:
                out *= a
            return out
        return float(e.evalf())

    for node in sp.preorder_traversal(expr):
        if getattr(node, "is_Pow", False) and node.exp.is_number:
            value(node.exp)
            if node.base.is_number:
                value(node)


def parse_equation(text: str, target: str = "u", indep: list[str] | None = None,
                   known: dict[str, float] | None = None) -> Equation:
    check_text(text)
    check_name(target, "target")
    for v in indep or []:
        check_name(v, "variable")
    known = dict(known or {})
    for k, v in known.items():
        try:
            known[k] = float(v)
        except (TypeError, ValueError):
            raise EquationError(f"the known value for '{k}' must be a number, not {v!r}") from None
    if text.count("=") != 1 or "==" in text:
        raise EquationError("write the equation with exactly one '=' sign, e.g. \"u_tt + k*u = 0\"")
    src = text.strip()

    # u'' -> u_<var><var> (only when there is one independent variable)
    primes = re.findall(rf"\b{re.escape(target)}('+)", src)
    if primes:
        if indep is not None and len(indep) != 1:
            raise EquationError("primes (u') need exactly one independent variable; use u_t, u_xx instead")
        var = indep[0] if indep else "t"
        if len(var) != 1:
            raise EquationError(f"{target}' needs a one-letter variable name, but the variable is "
                                f"'{var}'; rename that column to one letter (e.g. t)")
        src = re.sub(rf"\b{re.escape(target)}('+)", lambda m: f"{target}_{var * len(m.group(1))}", src)
        indep = indep or [var]

    if "'" in src:   # a quote anywhere but after the target would start a Python string
        raise EquationError(f"' is only allowed right after {target} (as in {target}' or {target}'')")

    # Derivative shorthand u_<letters>
    raw = re.findall(rf"\b{re.escape(target)}_([A-Za-z]+)\b", src)
    letters = []
    for suffix in raw:
        for ch in suffix:
            if ch not in letters:
                letters.append(ch)
    if indep is None:
        # keep a stable, readable order: x, y, z first, t last, others alphabetical
        rank = {"x": 0, "y": 1, "z": 2}
        indep = sorted(letters, key=lambda c: (c == "t", rank.get(c, 3), c)) or ["t"]
    for ch in letters:
        if ch not in indep:
            raise EquationError(
                (f"variable names in derivatives are one letter each, so '{target}_...' cannot "
                 f"refer to {', '.join(repr(v) for v in indep if len(v) > 1)}; rename those columns "
                 f"to one letter. " if any(len(v) > 1 for v in indep) else "") +
                f"'{target}_...' uses '{ch}', which is not an independent variable "
                f"(variables: {', '.join(indep)}). Each letter after '_' must be one variable.")
    order = {v: i for i, v in enumerate(indep)}

    derivs: dict[str, tuple[str, ...]] = {}

    def canon(m: re.Match) -> str:
        seq = tuple(sorted(m.group(1), key=lambda c: order[c]))
        name = _deriv_name(target, seq)
        derivs[name] = seq
        return name

    src = re.sub(rf"\b{re.escape(target)}_([A-Za-z]+)\b", canon, src)

    # Build the name table: every identifier is a symbol unless it is a known math
    # function / constant, or an unknown function (a name directly followed by "(").
    local: dict[str, object] = {}
    unknown_func_names: set[str] = set()
    for m in re.finditer(r"[A-Za-z_]\w*", src):
        name = m.group(0)
        if keyword.iskeyword(name):
            raise EquationError(f"'{name}' is a reserved Python word; rename it (e.g. '{name}_')")
        called = src[m.end():].startswith("(")      # f(u), not f (u): that is k*(...) missing '*' 
        if name in MATH_FUNCS:
            local[name] = MATH_FUNCS[name]
        elif name in CONSTANTS and not called:
            local[name] = CONSTANTS[name]
        elif called:
            if name == target or name in derivs or name in indep:
                raise EquationError(f"'{name}(...)' looks like a function call; write {name} on its own")
            if name in UNSUPPORTED_FUNCS:
                raise EquationError(f"'{name}(...)' is not supported in this version (only smooth "
                                    "functions of one argument); otherwise it would be learned as an "
                                    "unknown function. Rename it if you mean a function to learn")
            unknown_func_names.add(name)
            local[name] = sp.Function(name)
        else:
            local[name] = sp.Symbol(name)

    lhs_s, rhs_s = src.split("=")
    try:
        tr = standard_transformations + (convert_xor,)
        raw_sides = [parse_expr(s, local_dict=local, transformations=tr, evaluate=False)
                     for s in (lhs_s, rhs_s)]
    except EquationError:
        raise
    except RecursionError:
        raise EquationError("the equation is too deeply nested or too long to read") from None
    except Exception as exc:  # sympy raises many types here
        raise EquationError(f"could not read the equation: {_syntax_hint(text, exc)}") from None
    for side in raw_sides:
        _check_powers(side)
    try:     # exponents are now known to be small, so evaluating is safe
        lhs, rhs = (parse_expr(s, local_dict=local, transformations=tr, evaluate=True)
                    for s in (lhs_s, rhs_s))
    except RecursionError:
        raise EquationError("the equation is too deeply nested or too long to read") from None
    except Exception as exc:
        raise EquationError(f"could not read the equation: {_syntax_hint(text, exc)}") from None
    residual = lhs - rhs

    # Unknown functions: one argument each, replaced by placeholder symbols.
    ufuncs: list[UnknownFunction] = []
    apps = sorted(residual.atoms(sp.core.function.AppliedUndef), key=str)
    for i, app in enumerate(apps):
        if len(app.args) != 1:
            raise EquationError(f"{app}: unknown functions take one argument in this version")
        ph = f"_{app.func.__name__}_{i}"     # user names cannot start with _, so no clash
        ufuncs.append(UnknownFunction(ph, app.func.__name__, app.args[0]))
        residual = residual.subs(app, sp.Symbol(ph))

    reserved = {target, *derivs, *indep, *(u.placeholder for u in ufuncs)}
    free = sorted(str(s) for s in residual.free_symbols if str(s) not in reserved)
    for u in ufuncs:
        free += [str(s) for s in u.arg.free_symbols if str(s) not in reserved and str(s) not in free]
    for k in known:
        if k not in free:
            raise EquationError(f"known value given for '{k}', which is not in the equation")
    subs = {sp.Symbol(k): sp.Float(v) for k, v in known.items()}
    residual = residual.subs(subs)
    for u in ufuncs:
        u.arg = u.arg.subs(subs)
    params = [p for p in free if p not in known]

    if not derivs:
        raise EquationError(f"the equation has no derivative of {target} (write e.g. {target}_t or {target}')")
    if not any(residual.has(sp.Symbol(d)) for d in derivs):
        raise EquationError(f"the derivatives of {target} cancel out; nothing left to fit")

    return Equation(text=text, target=target, indep=list(indep), residual=residual,
                    derivs=dict(sorted(derivs.items(), key=lambda kv: (len(kv[1]), kv[0]))),
                    params=params, known=known, ufuncs=ufuncs)
