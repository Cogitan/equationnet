"""The candidate models and how each one's script is rendered from an equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import templates as T
from .codegen import explicit_function, residual_function
from .parse import Equation, EquationError

VERSION = "0.0.1"


class NotApplicable(Exception):
    """This model cannot handle this equation. The message says why, in words."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    doc: str
    uses_equation: bool
    needs: str
    render_body: Callable[[Equation, dict], str]


def _time_var(eq: Equation) -> str:
    return eq.time


def _fits(eq: Equation, model: "ModelSpec") -> str:
    if not model.uses_equation:
        return f"{eq.target}({', '.join(eq.indep)}) from the data alone (the equation is not used)"
    bits = []
    if eq.params:
        bits.append("constants " + ", ".join(eq.params))
    if eq.ufuncs:
        bits.append("unknown functions " + ", ".join(f"{u.name}({u.arg})" for u in eq.ufuncs))
    bits.append(f"{eq.target}({', '.join(eq.indep)})")
    return "; ".join(bits)


# ---- classical ------------------------------------------------------------

def _body_equation_fit(eq: Equation, cfg: dict) -> str:
    try:
        if len(eq.indep) == 1:
            t = eq.indep[0]
            order = eq.order_in(t)
            pure = {f"{eq.target}_{t * k}" for k in range(1, order + 1)}
            if set(eq.derivs) - pure:
                raise NotApplicable("uses derivatives this solver does not track")
            args = [t, eq.target] + [f"{eq.target}_{t * k}" for k in range(1, order)]
            fn = explicit_function(eq, "highest", f"{eq.target}_{t * order}", args)
            return T.EQUATION_FIT_ODE.replace("<<ORDER>>", str(order)).replace("<<EXPLICIT_FN>>", fn)
        if len(eq.indep) == 2:
            tv = _time_var(eq)
            xv = [v for v in eq.indep if v != tv][0]
            ut, ux, uxx = (f"{eq.target}_{s}" for s in (tv, xv, xv + xv))
            if eq.order_in(tv) != 1 or set(eq.derivs) - {ut, ux, uxx}:
                raise NotApplicable(
                    f"the classical solver handles {ut} = (terms in {eq.target}, {ux}, {uxx}) only; "
                    f"this equation has {', '.join(eq.derivs)}")
            fn = explicit_function(eq, "rhs", ut, [xv, tv, eq.target, ux, uxx])
            return (T.EQUATION_FIT_PDE.replace("<<XI>>", str(eq.indep.index(xv)))
                    .replace("<<TI>>", str(eq.indep.index(tv))).replace("<<EXPLICIT_FN>>", fn))
        raise NotApplicable("the classical solver handles one variable, or one space variable plus time")
    except EquationError as exc:
        raise NotApplicable(str(exc)) from None


# ---- physics-informed -----------------------------------------------------

def _physics(eq: Equation, cfg: dict, body: str, steps: int) -> str:
    derivs = "{" + ", ".join(f'"{k}": {tuple(v)!r}' for k, v in eq.derivs.items()) + "}"
    d = len(eq.indep)
    return (T.PHYSICS.replace("<<DERIVS>>", derivs)
            .replace("<<NCOL>>", str(cfg.get("collocation", 512 if d == 1 else 2000)))
            .replace("<<STEPS>>", str(cfg.get("steps", steps)))
            .replace("<<LBFGS>>", str(cfg.get("lbfgs", 300)))
            .replace("<<RESIDUAL_FN>>", residual_function(eq, "torch"))
            .replace("<<BODY>>", body))


def _body_pinn(eq: Equation, cfg: dict) -> str:
    # 1000 Adam steps: in the headroom test 500 to 3000 gave the same accuracy (the
    # L-BFGS finish does the work) and 1000 leaves margin for harder problems
    return _physics(eq, cfg, T.BODY_MLP, 1000)


def _cheb_degree(d: int) -> int:
    return {1: 24, 2: 10, 3: 5}.get(d, 3)


def _body_poly_physics(eq: Equation, cfg: dict) -> str:
    deg = cfg.get("degree", _cheb_degree(len(eq.indep)))
    # the polynomial's steps are cheap (fixed derivative matrices), so it keeps 3000
    return _physics(eq, cfg, T.BODY_CHEB.replace("<<DEG>>", str(deg)), 3000)


# ---- data only ------------------------------------------------------------

def _body_nn(eq: Equation, cfg: dict) -> str:
    return (T.NN_DATA.replace("<<STEPS>>", str(cfg.get("steps", 3000)))
            .replace("<<LBFGS>>", str(cfg.get("lbfgs", 300))))


def _body_spline(eq: Equation, cfg: dict) -> str:
    return T.SPLINE_DATA


def _body_poly(eq: Equation, cfg: dict) -> str:
    return T.POLY_DATA.replace("<<MAXDEG>>", str({1: 15, 2: 10}.get(len(eq.indep), 4)))


MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    ModelSpec("equation_fit", "Equation fit (classical)",
              "Solves the equation with a standard numerical solver (SciPy) and fits the unknowns\n"
              "by least squares. No neural network. Unknown functions are cubic polynomials.",
              True, "numpy, scipy", _body_equation_fit),
    ModelSpec("pinn", "PINN (neural network + equation)",
              "Physics-informed neural network: a small network trained so it matches the data\n"
              "and satisfies the equation. Unknowns are trained alongside it.",
              True, "numpy, torch", _body_pinn),
    ModelSpec("poly_physics", "Polynomial + equation",
              "Chebyshev polynomial trained like a PINN: matches the data and satisfies the equation.",
              True, "numpy, torch", _body_poly_physics),
    ModelSpec("nn", "Neural network (data only)",
              "Plain neural network fitted to the data. Ignores the equation: a baseline.",
              False, "numpy, torch", _body_nn),
    ModelSpec("spline", "Smoothing spline (data only)",
              "Smoothing spline through the data (cubic in 1D, thin-plate in 2D+), smoothness\n"
              "picked automatically. Ignores the equation: the strongest simple baseline.",
              False, "numpy, scipy", _body_spline),
    ModelSpec("poly", "Polynomial (data only)",
              "Chebyshev polynomial least squares, degree picked by cross-validation.\n"
              "Ignores the equation: a baseline.",
              False, "numpy", _body_poly),
]}


class _Settings(dict):
    """Per-model settings that remember which keys the model read."""

    def __init__(self, *a):
        super().__init__(*a)
        self.read = set()

    LIMITS = {"steps": 0, "lbfgs": 0, "collocation": 1, "degree": 1}

    def get(self, key, default=None):
        self.read.add(key)
        value = super().get(key, default)
        if key in self and key in self.LIMITS:
            if isinstance(value, bool) or not isinstance(value, int) or value < self.LIMITS[key]:
                raise ValueError(f"setting '{key}' must be a whole number of at least "
                                 f"{self.LIMITS[key]}, not {value!r}")
        return value

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def render(eq: Equation, model: str, cfg: dict | None = None) -> str:
    """Full standalone script for one model. Raises NotApplicable with a reason."""
    if model not in MODELS:
        raise KeyError(f"unknown model '{model}'; choose from {', '.join(MODELS)}")
    spec = MODELS[model]
    cfg = _Settings(cfg or {})
    body = spec.render_body(eq, cfg)
    unused = [k for k in cfg if k not in cfg.read]
    if unused:   # never ignore a setting silently
        raise ValueError(f"the {spec.title} model does not use the setting(s) "
                         f"{', '.join(map(repr, unused))}")
    uses = spec.uses_equation
    header = (T.HEADER.replace("<<VERSION>>", VERSION).replace("<<MODEL>>", spec.name)
              .replace("<<MODEL_DOC>>", spec.doc).replace("<<EQUATION>>", eq.text)
              .replace("<<FITS>>", _fits(eq, spec))
              .replace("<<COLUMNS>>", ", ".join(eq.indep + [eq.target]))
              .replace("<<NEEDS>>", spec.needs)
              .replace("<<INDEP>>", repr(eq.indep)).replace("<<TARGET>>", eq.target)
              .replace("<<PARAMS>>", repr(eq.params if uses else []))
              .replace("<<UFUNCS>>", repr(eq.ufunc_names if uses else [])))
    return header + body + T.FOOTER.replace("<<MODEL>>", spec.name)
