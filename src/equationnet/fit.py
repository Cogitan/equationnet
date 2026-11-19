"""fit(): build every model for an equation, train each, rank them on held-out data."""

from __future__ import annotations

import csv
import inspect
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import MODELS, NotApplicable, render
from .parse import Equation, check_name, parse_equation


@dataclass
class Candidate:
    name: str
    title: str
    uses_equation: bool
    script: str | None = None
    status: str = "ok"                 # "ok", "not applicable: ...", "failed: ..."
    holdout_rmse: float = float("nan")   # mean over folds when cross-validated
    train_rmse: float = float("nan")
    fold_rmse: list = field(default_factory=list)
    fold_errors: list = field(default_factory=list, repr=False)   # held-out residuals per fold
    params: dict = field(default_factory=dict)
    param_se: dict = field(default_factory=dict)    # standard errors, where the model gives them
    seconds: float = 0.0
    namespace: dict | None = field(default=None, repr=False)
    state: object = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def holdout_errors(self) -> np.ndarray | None:
        """Residuals on every held-out point, all folds together."""
        return np.concatenate(self.fold_errors) if self.fold_errors else None

    @property
    def note(self) -> str | None:
        """A warning the model attached to its fit (for example, another answer fits the
        data equally well), or None."""
        return self.state.get("note") if isinstance(self.state, dict) else None

    def predict(self, X) -> np.ndarray:
        return self.namespace["predict"](self.state, np.asarray(X, float))

    def learned_function(self, name: str, z) -> np.ndarray:
        return self.namespace["learned_function"](self.state, name, z)

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.script, encoding="utf-8")
        return path


# Tie rule. When several models are within the noise of the lowest held-out error,
# the held-out data cannot say which is better, so the one the equation constrains most
# is returned: the more the equation constrains a model, the less it can fit noise.
# Order: the classical fit (a few constants), the physics polynomial (tens of
# coefficients, held by the equation), the PINN (thousands of weights, held by the
# equation), then the data-only models.
SIMPLEST_FIRST = ["equation_fit", "poly_physics", "pinn", "poly", "spline", "nn"]


def _with_se(name, value, se):
    if se is None:
        return f"{name}={value:.4g}"
    if not np.isfinite(se):
        return f"{name}={value:.4g} (+-inf)"
    return f"{name}={value:.4g}+-{se:.2g}"


TIE_SE = 2.0   # how many standard errors of the best model's held-out MSE count as a tie


def _within_noise(c: "Candidate", top: "Candidate") -> bool:
    """Two-standard-error rule: c ties with top when its mean squared held-out error is
    within TIE_SE standard errors of top's (the standard error of top's own MSE).

    Measured against the best model's own spread, not the paired difference: a paired
    test can call a model with one exploding prediction "tied" (its huge error inflates
    the spread it is judged by), and can leave out a close model while tying a worse one.
    Chosen in the stage F review (TESTPLAN.md) over the paired one-standard-error rule."""
    a, b = c.holdout_errors, top.holdout_errors
    if a is None or b is None or len(a) != len(b) or len(a) < 2:
        return False
    se = float(np.std(b ** 2, ddof=1) / np.sqrt(len(b)))
    return float(np.mean(a ** 2)) <= float(np.mean(b ** 2)) + TIE_SE * se


NOISE_QUANTILE = 0.975   # a model is also tied when its held-out MSE is below this quantile
                         # of what pure measurement noise gives on that many points


def _noise_floor(candidates, n: int) -> float | None:
    """Largest held-out MSE that pure measurement noise reaches on n points (97.5% of the
    time), or None. The noise level is the smallest training error of the data-only
    models (they assume no equation, so a wrong equation cannot raise it; a training
    error is at or below the noise, so this errs toward fewer ties).

    Why: with a few held-out points, a data-only model can score well below the noise
    by luck while a model that is right to within the noise scores at the noise, and
    the standard-error test cannot see that both are at the noise floor."""
    from scipy.stats import chi2
    s = [c.train_rmse for c in candidates if c.ok and not c.uses_equation and np.isfinite(c.train_rmse)]
    if not s or n < 1:
        return None
    return float(chi2.ppf(NOISE_QUANTILE, n) / n * min(s) ** 2)


@dataclass
class Result:
    equation: Equation
    split: str
    n_train: int                       # points the returned models were trained on
    n_holdout: int                     # points scored per fold (or in the single split)
    candidates: dict[str, Candidate]
    folds: int = 1                     # 1 = one held-out split; >1 = cross-validation
    pick: str = "simplest"             # "simplest" (tie rule) or "lowest" (lowest error)
    dropped: list = field(default_factory=list)   # data rows left out (missing or non-finite values)
    notes: list = field(default_factory=list)     # other things the result must say

    @property
    def ranked(self) -> list[Candidate]:
        ok = [c for c in self.candidates.values() if c.ok and np.isfinite(c.holdout_rmse)]
        return sorted(ok, key=lambda c: c.holdout_rmse)

    @property
    def tied(self) -> list[Candidate]:
        """Models within the noise of the lowest held-out error (lowest first): within two
        standard errors of it, or at the measurement-noise floor (_noise_floor)."""
        r = self.ranked
        if not r:
            return []
        e = r[0].holdout_errors
        floor = _noise_floor(self.candidates.values(), len(e)) if e is not None else None
        return r[:1] + [c for c in r[1:] if _within_noise(c, r[0]) or (
            floor is not None and c.holdout_errors is not None and float(np.mean(c.holdout_errors ** 2)) <= floor)]

    @property
    def best(self) -> Candidate:
        """pick="simplest": the simplest model among those tied with the lowest error.
        pick="lowest": the lowest error."""
        if not self.ranked:
            raise RuntimeError("no model finished; see result.table()")
        if self.pick == "lowest":
            return self.ranked[0]
        order = lambda c: SIMPLEST_FIRST.index(c.name) if c.name in SIMPLEST_FIRST else len(SIMPLEST_FIRST)
        return min(self.tied, key=order)

    def __getitem__(self, name: str) -> Candidate:
        return self.candidates[name]

    def _constant_checks(self):
        """Per constant: (data-based problems, model disagreement or None, classical value,
        classical standard error)."""
        good = [c for c in self.tied if c.uses_equation and c.params]
        ref = self.candidates.get("equation_fit")
        out = {}
        for p in self.equation.params:
            pool = good + ([ref] if ref is not None and ref.ok and ref not in good else [])
            vals = {c.title: c.params[p] for c in pool if p in c.params and np.isfinite(c.params[p])}
            se = ref.param_se.get(p) if ref is not None and ref.ok else None
            v = ref.params.get(p) if ref is not None and ref.ok else None
            parts = []
            if se is not None and not np.isfinite(se):
                parts.append(f"not determined (the classical fit is as good along a whole "
                             f"range of {p})")
            elif se is not None and v is not None and abs(v) <= 2 * se:
                parts.append(f"consistent with 0 (classical fit {v:.4g} +- {se:.2g}): the data "
                             "cannot tell it from zero")
            elif se is not None and v is not None and se > 0.5 * abs(v):
                parts.append(f"poorly determined (classical fit {v:.4g} +- {se:.2g})")
            elif ref is not None and ref.ok and isinstance(ref.state, dict) and p in ref.state.get("loose", ()):
                parts.append(f"poorly determined: set to half or one and a half times its value "
                             f"({v:.4g}), the classical fit refits the data as well")
            spread = None
            if len(vals) > 1:
                lo, hi = min(vals.values()), max(vals.values())
                big = hi - lo > 0.1 * max(abs(lo), abs(hi))
                if se is not None and np.isfinite(se):
                    big = big and hi - lo > 2 * se
                if big:
                    spread = (lo, hi)
            out[p] = (parts, spread, v, se)
        return out

    def constant_notes(self) -> list[str]:
        """Plain-language warnings about constants the data do not pin down.

        Judged from the classical fit's standard error: infinite, more than half the
        value, or the value within 2 standard errors of zero; or from its profile check
        (the classical ODE fit refits the data as well with the constant moved by half). Disagreement between the
        equation models is added here only when there is no such standard error, or the
        constant is already flagged; otherwise it is a separate note
        (disagreement_notes), because a pinned-down classical fit and a looser PINN can
        disagree while the data do determine the constant."""
        notes = []
        for p, (parts, spread, v, se) in self._constant_checks().items():
            if spread and (parts or se is None or not np.isfinite(se)):
                parts = parts + [f"the equation models disagree, {spread[0]:.4g} to {spread[1]:.4g}"]
            if parts:
                notes.append(f"{p}: " + "; ".join(parts) + ". If you know it, pass known={'" + p +
                             "': ...}. Otherwise more or longer data may help, unless only a "
                             "combination of constants is determined.")
        return notes

    def disagreement_notes(self) -> list[str]:
        """Constants the classical fit pins down (small standard error) but on which the
        equation models the held-out data cannot tell apart give clearly different values
        (more than 10% and more than 2 standard errors apart)."""
        notes = []
        for p, (parts, spread, v, se) in self._constant_checks().items():
            if spread and not parts and se is not None and np.isfinite(se):
                notes.append(f"{p}: the equation models give {spread[0]:.4g} to {spread[1]:.4g}, "
                             f"while the classical fit gives {v:.4g} +- {se:.2g}. The data pin {p} "
                             "down if the classical fit is right; the models' predictions are "
                             "equally good on the held-out points, so compare them on your data "
                             "before trusting one value.")
        return notes

    def table(self) -> str:
        eq = self.equation
        if self.folds > 1:
            how = (f"{self.folds}-fold cross-validation" if self.split == "random" else
                   f"{self.folds} forecast origins (train on the past, score what comes next)")
            split_line = (f"split: {self.split}, {how}; errors are the mean over folds.\n"
                          f"       Returned models are retrained on all {self.n_train} points.")
        else:
            split_line = f"split: {self.split}, {self.n_train} points to train, {self.n_holdout} held out"
        lines = [f"equation: {eq.text}", f"          {eq.describe()}", split_line]
        if self.dropped:
            shown = ", ".join(map(str, self.dropped[:10])) + (", ..." if len(self.dropped) > 10 else "")
            lines.append(f"left out: {len(self.dropped)} row{'s' if len(self.dropped) > 1 else ''} with a "
                         f"missing or non-finite value (data rows {shown})")
        lines.append("")
        head = f"{'rank':<5}{'model':<34}{'held-out error':>15}{'time':>8}   fitted"
        lines += [head, "-" * len(head)]
        rank = {c.name: i + 1 for i, c in enumerate(self.ranked)}
        order = self.ranked + [c for c in self.candidates.values() if c.name not in rank]
        tied = {c.name for c in self.tied} if len(self.tied) > 1 else set()
        for c in order:
            if c.name in rank:
                fitted = ", ".join(_with_se(k, v, c.param_se.get(k)) for k, v in c.params.items()) or "-"
                mark = "=" if c.name in tied else " "
                lines.append(f"{rank[c.name]:<3}{mark} {c.title:<34}{c.holdout_rmse:>15.4g}"
                             f"{c.seconds:>7.1f}s   {fitted}")
            else:
                lines.append(f"{'-':<5}{c.title:<34}   {c.status}")
        if self.ranked:
            lines += ["", f"best: {self.best.title}. Held-out error is the RMS error on data the models never saw."]
            if tied:
                n = len(self.ranked[0].holdout_errors)
                lines.append(f"=     {len(tied)} models are within the noise of the lowest error on these "
                             f"{n} held-out points, so the data cannot tell them apart.")
                if self.pick == "lowest":
                    lines.append('      pick="lowest": the lowest error is returned anyway.')
                elif self.best is self.ranked[0]:
                    lines.append("      The lowest is also the one the equation constrains most.")
                else:
                    lines.append(f"      The one the equation constrains most is returned; pick=\"lowest\" "
                                 f"would return {self.ranked[0].title}.")
        lines += [f"note ({c.title}): {c.note}" for c in order if c.ok and c.note]
        lines += [f"note: {n}" for n in self.notes]
        if any(c.param_se for c in self.ranked):
            lines.append("+-: one standard error, if the equation is exactly right (a wrong or "
                         "approximate equation can be off by more).")
        lines += [f"not pinned down: {n}" for n in self.constant_notes()]
        lines += [f"models disagree: {n}" for n in self.disagreement_notes()]
        return "\n".join(lines)

    def save_scripts(self, folder) -> list[Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        return [c.save(folder / f"{c.name}.py") for c in self.candidates.values() if c.script]


MIN_POINTS = 5     # fewest data points a fit is attempted on
_MISSING = {"", "nan", "na", "n/a", "null", "none", "-"}


def _number(cell: str, row: int, col: str) -> float:
    c = cell.strip()
    if c.lower() in _MISSING:
        return float("nan")
    try:
        return float(c)
    except ValueError:
        raise ValueError(f"data row {row}, column '{col}': {c!r} is not a number") from None


def _read_csv(path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:   # -sig: Excel's byte-order mark
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]   # blank lines hold nothing
    if not rows:
        raise ValueError(f"{path.name} is empty")
    head = rows[0]
    # Excel often adds empty unnamed columns at the right: they hold nothing, so they go;
    # an unnamed column WITH values cannot be used and is named by its position
    keep = []
    for j, h in enumerate(head):
        if h.strip():
            keep.append(j)
        elif any(j < len(r) and r[j].strip() for r in rows[1:]):
            raise ValueError(f"column {j + 1} of {path.name} has values but no name in the header")
    if len(keep) < len(head):
        rows = [[r[j] if j < len(r) else "" for j in keep] for r in rows]
        head = rows[0]
    if len(head) == 1:
        for sep, word in ((";", "semicolons"), ("\t", "tabs"), ("|", "'|'")):
            if sep in head[0]:
                raise ValueError(f"{path.name} seems to separate values with {word}; "
                                 "equationnet reads comma-separated files (save as CSV with commas)")
    names = [h.strip() for h in head]
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name} has a repeated column name: {', '.join(names)}")
    body = rows[1:]
    if not body:
        raise ValueError(f"{path.name} has a header but no data rows (need at least {MIN_POINTS} points)")
    cols = {n: [] for n in names}
    for i, r in enumerate(body, start=1):
        if len(r) != len(names):
            raise ValueError(f"data row {i} of {path.name} has {len(r)} values; the header has {len(names)}")
        for n, cell in zip(names, r):
            cols[n].append(_number(cell, i, n))
    out = {n: np.asarray(v, float) for n, v in cols.items()}
    empty = [n for n, v in out.items() if not np.isfinite(v).any()]
    if empty:
        raise ValueError(f"column(s) {', '.join(map(repr, empty))} of {path.name} have no values; "
                         "remove them")
    return out


def _columns_from(data) -> dict[str, np.ndarray]:
    if isinstance(data, (str, Path)):
        return _read_csv(data)
    items = [(str(c), data[c]) for c in data.columns] if hasattr(data, "columns") else \
        [(str(k), v) for k, v in dict(data).items()]
    cols = {}
    for name, v in items:
        try:
            cols[name] = np.asarray(v, float).ravel()
        except (TypeError, ValueError):
            raise ValueError(f"column '{name}' contains values that are not numbers") from None
    lengths = {n: len(v) for n, v in cols.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError("the columns have different lengths: " +
                         ", ".join(f"{n} {k}" for n, k in lengths.items()))
    return cols


def _load_data(data, target: str) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    """Accepts a CSV path, a dict of columns, or a pandas DataFrame. Rows with a missing
    or non-finite value are left out and their (1-based) row numbers returned, so the
    result can say so; nothing else is changed or dropped."""
    cols = _columns_from(data)
    check_name(target, "target")
    if target not in cols:
        raise ValueError(f"the data has no column '{target}' (columns: {', '.join(cols)}); "
                         "pass target=... if it has another name")
    for name in cols:
        check_name(name, "column")
    indep = [c for c in cols if c != target]
    if not indep:
        raise ValueError(f"the data needs at least one variable column besides '{target}' (e.g. t)")
    X = np.column_stack([cols[c] for c in indep])
    y = cols[target]
    keep = np.all(np.isfinite(X), 1) & np.isfinite(y)
    dropped = [int(i) + 1 for i in np.flatnonzero(~keep)]
    if keep.sum() < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} data points with every value present; "
                         f"got {int(keep.sum())}")
    return indep, X[keep], y[keep], dropped


def _tcol(indep, time=None):
    if time is not None:
        return indep.index(time)
    return indep.index("t") if "t" in indep else len(indep) - 1


def _time_slices(t, holdout):
    """Distinct times, and how many of the last ones a forecast holds out. Counting
    distinct times (not points) keeps whole time slices together on gridded data,
    where every time repeats once per position."""
    ut = np.unique(t)
    if len(ut) < 2:
        raise ValueError("a forecast split needs at least 2 distinct times")
    return ut, max(1, int(round(holdout * len(ut))))


def _check_train(train, what):
    if train.sum() < 3:
        raise ValueError(f"{what} leaves {int(train.sum())} point(s) to train on; at least 3 are "
                         "needed (use a smaller holdout or fewer folds)")


def _split(X, y, indep, split, holdout, seed, time=None):
    """One held-out split. Returns (train mask, test mask)."""
    n = len(y)
    n_out = max(1, int(round(holdout * n)))
    if split == "random":
        idx = np.random.default_rng(seed).permutation(n)
        test = np.zeros(n, bool)
        test[idx[:n_out]] = True
    elif split == "forecast":
        t = X[:, _tcol(indep, time)]
        ut, k_out = _time_slices(t, holdout)
        test = t >= ut[-k_out]
    else:
        raise ValueError("split must be 'random' or 'forecast'")
    _check_train(~test, f"holdout={holdout} with split='{split}'")
    return ~test, test


def _folds(X, y, indep, split, holdout, seed, k, time=None):
    """Cross-validation folds as a list of (train mask, test mask).

    random   : k-fold, every point scored once.
    forecast : k forecast origins. Each trains on everything before its window and
               scores the next `holdout` share of the distinct times; the first origin
               is the ordinary forecast split and each earlier one steps back by a whole
               window, so no point is scored twice."""
    n = len(y)
    if split == "random":
        if k > n:
            raise ValueError(f"cv={k} folds need at least {k} points; got {n}")
        idx = np.random.default_rng(seed).permutation(n)
        out = []
        for part in np.array_split(idx, k):
            test = np.zeros(n, bool)
            test[part] = True
            _check_train(~test, f"cv={k}")
            out.append((~test, test))
        return out
    if split == "forecast":
        t = X[:, _tcol(indep, time)]
        ut, k_out = _time_slices(t, holdout)
        out = []
        for j in range(k):
            end = len(ut) - j * k_out
            start = end - k_out
            if start < 2:
                raise ValueError(f"cv={k} forecast origins of {k_out} time(s) each need more than "
                                 f"{k * k_out + 1} distinct times; there are {len(ut)}. Use fewer "
                                 "folds or a smaller holdout")
            test = (t >= ut[start]) & (t <= ut[end - 1])
            train = t < ut[start]
            _check_train(train, f"cv={k} forecast origin {j + 1}")
            out.append((train, test))
        return out
    raise ValueError("split must be 'random' or 'forecast'")


SMALL_DATA = 50   # below this many points, cv="auto" cross-validates

# Models run in this fixed order, each printing its result as it finishes: the
# quick data-only baselines first, then the equation models (the data-only network
# runs between the physics polynomial and the PINN).
RUN_ORDER = ["poly", "spline", "equation_fit", "poly_physics", "nn", "pinn"]


def _n_folds(cv, split, n):
    if cv == "auto":
        return (5 if split == "random" else 3) if n < SMALL_DATA else 1
    if isinstance(cv, bool) or not isinstance(cv, (int, np.integer)):
        raise ValueError(f"cv must be 'auto' or a whole number of folds (1 = one held-out split), "
                         f"not {cv!r}")
    k = int(cv)
    if k < 1:
        raise ValueError("cv must be 'auto' or a whole number of folds (1 = one held-out split)")
    return k


def _as_points(at, indep, with_extra=False):
    """predict_at as an (n, len(indep)) array, from an array, dict, DataFrame or CSV path.
    with_extra: also return the names of columns that are not variables (ignored)."""
    extra = []
    if isinstance(at, (str, Path)):
        at = _read_csv(at)
    if hasattr(at, "columns") or isinstance(at, dict):
        missing = [v for v in indep if v not in at]
        if missing:
            raise ValueError(f"predict_at has no column {', '.join(missing)} "
                             f"(it needs the data's variables: {', '.join(indep)})")
        a = np.column_stack([np.asarray(at[v], float) for v in indep])
        extra = [str(c) for c in (at.columns if hasattr(at, "columns") else at) if c not in indep]
    else:
        a = np.asarray(at, float)
        a = a.reshape(-1, 1) if a.ndim == 1 and len(indep) == 1 else np.atleast_2d(a)
        if a.shape[1] != len(indep):
            raise ValueError(f"predict_at needs {len(indep)} columns ({', '.join(indep)})")
    if not np.all(np.isfinite(a)):
        raise ValueError("predict_at contains a missing or non-finite value")
    return (a, extra) if with_extra else a


def _run_script(script: str, name: str) -> dict:
    ns: dict = {"__name__": f"equationnet_{name}"}
    exec(compile(script, f"<equationnet:{name}>", "exec"), ns)
    return ns


def _train_eval(script, name, X, y, train, test, seed, domain):
    """Run a model script, train on `train`; returns (namespace, state, train rmse,
    test rmse, test residuals)."""
    ns = _run_script(script, name)
    kwargs = {"seed": seed}
    if "domain" in inspect.signature(ns["train"]).parameters:
        # enforce the equation wherever predictions are wanted (locations only,
        # never the held-out values)
        kwargs["domain"] = domain
    state = ns["train"](X[train], y[train], **kwargs)

    def resid(m):
        return ns["predict"](state, X[m]) - y[m]

    def rmse(r):
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")

    r_test = resid(test)
    return ns, state, rmse(resid(train)), rmse(r_test), r_test


def fit(equation: str, data, target: str = "u", known: dict | None = None,
        models: list[str] | str = "all", split: str = "random", holdout: float = 0.2,
        seed: int = 0, cv: int | str = 1, predict_at=None, config: dict | None = None,
        pick: str = "simplest", verbose: bool = True) -> Result:
    """Train every applicable model and rank them by error on held-out data.

    equation   : text, e.g. "u_tt + c*u_t + k*u = 0"
    data       : CSV path, dict of columns, or DataFrame; one column per variable plus the target
    known      : values for constants you already know, e.g. {"k": 4.0}
    split      : "random" (hold out random points) or "forecast" (hold out the end in time)
    cv         : 1 (default): one held-out split of `holdout`; models are trained on the rest.
                 k > 1: k-fold cross-validation (forecast: k forecast origins), then every
                 model is retrained on all the data. "auto": cross-validate only under 50
                 points (5 folds random, 3 forecast origins). Every model is trained
                 once per fold and once more on all the data, so it is off by default.
    predict_at : locations where you will want predictions (e.g. future times); models
                 that use the equation also enforce it there. Same columns as the data.
    config     : per-model settings, e.g. {"pinn": {"steps": 1000}}
    pick       : "simplest" (default): among models within the noise of the lowest
                 held-out error (two standard errors of its mean squared error), return
                 the one the equation constrains most (classical fit, physics
                 polynomial, PINN, then the data-only models). "lowest": the lowest error.
    """
    if pick not in ("simplest", "lowest"):
        raise ValueError('pick must be "simplest" or "lowest"')
    if not isinstance(holdout, (int, float)) or not 0 < holdout < 1:
        raise ValueError(f"holdout is the share of points held out: a number between 0 and 1 "
                         f"(e.g. 0.2), not {holdout!r}")
    names = list(MODELS) if models == "all" else ([models] if isinstance(models, str) else list(models))
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {', '.join(map(repr, unknown))}; models: {', '.join(MODELS)}")
    unknown_cfg = [m for m in (config or {}) if m not in MODELS]
    if unknown_cfg:
        raise ValueError(f"config names unknown model(s) {', '.join(map(repr, unknown_cfg))}; "
                         f"models: {', '.join(MODELS)}")
    indep, X, y, dropped = _load_data(data, target)
    eq = parse_equation(equation, target=target, indep=indep, known=known)
    used = set().union(*eq.derivs.values()) | {str(s) for s in eq.residual.free_symbols}
    for u in eq.ufuncs:
        used |= {str(s) for s in u.arg.free_symbols}
    unused = [c for c in indep if c not in used]
    if unused:
        raise ValueError(f"column(s) {', '.join(map(repr, unused))} are not in the equation "
                         f"'{equation}'. If {target} depends on them the equation must say how; "
                         "otherwise remove them from the data")
    if len(y) - max(1, int(round(holdout * len(y)))) < 3:
        raise ValueError(f"holdout={holdout} leaves fewer than 3 of the {len(y)} points to train on")
    names.sort(key=lambda m: RUN_ORDER.index(m) if m in RUN_ORDER else len(RUN_ORDER))
    k = _n_folds(cv, split, len(y))
    if k == 1:
        folds = [_split(X, y, indep, split, holdout, seed, time=eq.time)]
        final = folds[0][0]                      # returned models: the training part
    else:
        folds = _folds(X, y, indep, split, holdout, seed, k, time=eq.time)
        final = np.ones(len(y), bool)            # returned models: all the data
    notes = []
    if predict_at is None:
        where = X
    else:
        pts, extra = _as_points(predict_at, indep, with_extra=True)
        if extra:
            notes.append(f"predict_at column(s) {', '.join(extra)} ignored: only the variables "
                         f"{', '.join(indep)} are locations")
        where = np.vstack([X, pts])
    domain = (where.min(0), where.max(0))
    config = config or {}
    unknown = [m for m in config if m not in MODELS]
    if unknown:
        raise ValueError(f"config names unknown model(s) {', '.join(map(repr, unknown))}; "
                         f"models: {', '.join(MODELS)}")
    for m, cfg in config.items():        # check settings even for models not run this time
        if m not in names:
            try:
                render(eq, m, cfg)
            except NotApplicable:
                pass
    out: dict[str, Candidate] = {}
    for name in names:
        spec = MODELS[name]
        cand = Candidate(name, spec.title, spec.uses_equation)
        out[name] = cand
        try:
            cand.script = render(eq, name, config.get(name))
            if "\nDOMAIN = None" in cand.script:
                # the saved script enforces the equation where it was enforced when scored:
                # the data's range plus predict_at (it used to say None)
                lows, highs = ([float(v) for v in d] for d in domain)
                cand.script = cand.script.replace(
                    "\nDOMAIN = None", f"\nDOMAIN = [{lows!r}, {highs!r}]", 1)
        except NotApplicable as exc:
            cand.status = f"not applicable: {exc}"
            if verbose:
                print(f"  {spec.title:<34} {cand.status}")
            continue
        t0 = time.perf_counter()
        errors = []
        try:
            for i, (tr, te) in enumerate(folds):
                try:
                    ns, state, rm_tr, rm_te, r_te = _train_eval(cand.script, name, X, y, tr, te, seed, domain)
                except Exception as exc:
                    if k > 1:
                        raise RuntimeError(f"in fold {i + 1} of {k}: {type(exc).__name__}: {exc}") from exc
                    raise
                cand.fold_rmse.append(rm_te)
                errors.append(r_te)
            if k > 1:
                ns, state, rm_tr, _, _ = _train_eval(cand.script, name, X, y, final, ~final, seed, domain)
            cand.namespace, cand.state = ns, state
            cand.train_rmse = rm_tr
            cand.holdout_rmse = float(np.mean(cand.fold_rmse))
            cand.fold_errors = errors
            cand.params = ns["fitted_params"](state)
            if "uncertainty" in ns:
                cand.param_se = ns["uncertainty"](state)
            if not np.isfinite(cand.holdout_rmse):
                cand.status = ("failed: the trained model predicts NaN or infinite values on held-out "
                               "points (training diverged; the equation may be undefined for some "
                               "values, e.g. log or sqrt of a negative number)")
        except Exception as exc:
            cand.status = f"failed: {type(exc).__name__}: {exc}"
            if verbose > 1:
                traceback.print_exc()
        cand.seconds = time.perf_counter() - t0
        if verbose:
            msg = f"{cand.holdout_rmse:.4g}" if cand.ok else cand.status
            print(f"  {spec.title:<34} {msg}  ({cand.seconds:.1f}s)")
    n_test = int(round(np.mean([te.sum() for _, te in folds])))
    scored = sum(int(te.sum()) for _, te in folds)
    if scored < 5:
        notes.append(f"the ranking rests on only {scored} held-out point(s); treat it as a rough "
                     "guide (more data, a larger holdout, or cv=5 scores more points)")
    return Result(eq, split, int(final.sum()), n_test, out, folds=k, pick=pick, dropped=dropped,
                  notes=notes)


def generate(equation: str, model: str = "pinn", target: str = "u", indep: list[str] | None = None,
             known: dict | None = None, config: dict | None = None) -> str:
    """Script for one model without any data: the training setup for this equation.
    config: this model's settings, e.g. {"steps": 1000}; the fit() form
    {"pinn": {"steps": 1000}} is accepted too."""
    eq = parse_equation(equation, target=target, indep=indep, known=known)
    if config and set(config) <= set(MODELS):
        others = [m for m in config if m != model]
        if others:
            raise ValueError(f"generate() writes one script ({model}); config also has settings for "
                             f"{', '.join(others)}, which would not be used")
        config = config.get(model, {})
    return render(eq, model, config)
