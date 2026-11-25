"""fit(): build every model for an equation, train each, rank them on held-out data."""

from __future__ import annotations

import copy
import csv
import inspect
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import MODELS, NotApplicable, render
from .parse import Equation, check_name, parse_equation


@dataclass(frozen=True)
class YourModel:
    """Your own model, scored on the same held-out points as the built-in ones.

    model         : anything with fit(X, y) and predict(X) that copy.deepcopy can copy (a
                    scikit-learn estimator, your own class), or a plain function predict(X) that needs no
                    training. X has one column per variable, in the data's column order
                    (target left out); y is the target. A fresh copy is trained for every
                    held-out split, so the object you pass is never changed.
    name          : a short name for the table (default: the class or function name;
                    your_model_N, by position, for a lambda or a callable with no name).
    uses_equation : whether the model uses the equation. False (default) treats it as a
                    data-only model. Either way, when it ties with a built-in model the
                    built-in one is returned: equationnet cannot see how much your model
                    is constrained.

    Optional methods on the model: fitted_params() -> {name: value} and
    uncertainty() -> {name: standard error}, shown in the table like the built-in models'.
    """
    model: object
    name: str | None = None
    uses_equation: bool = False


def custom(model, name: str | None = None, uses_equation: bool = False) -> YourModel:
    """Wrap your own model to name it or say that it uses the equation; see YourModel."""
    return YourModel(model, name, uses_equation)


@dataclass
class Candidate:
    name: str
    title: str
    uses_equation: bool
    script: str | None = field(default=None, repr=False)
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
    yours: bool = False                # a model passed in by the user (YourModel)

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
        if "learned_function" not in (self.namespace or {}):
            raise ValueError(f"{self.title} has no learned functions")
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


def _simplicity(c: "Candidate"):
    """Tie order: the built-in models in SIMPLEST_FIRST order, then yours (those that
    use the equation first). How much your model is constrained is unknown, so a tie
    never resolves to it while a built-in model is tied."""
    if not c.yours and c.name in SIMPLEST_FIRST:
        return (SIMPLEST_FIRST.index(c.name), 0)
    return (len(SIMPLEST_FIRST), 0 if c.uses_equation else 1)


def _noise_bound(e: np.ndarray) -> float:
    """Upper confidence bound on the noise standard deviation from n held-out residuals:
    rms * sqrt(n / chi2_0.025(n)), 97.5% for zero-mean normal residuals of one model chosen
    beforehand (somewhat less for the lowest of several models). If the model behind them
    is not at the noise, the true noise is lower still."""
    from scipy.stats import chi2
    n = len(e)
    return float(np.sqrt(np.mean(e ** 2)) * np.sqrt(n / chi2.ppf(0.025, n)))


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


def _noise_floor(candidates, n: int, cap: float | None = None) -> float | None:
    """Largest held-out MSE that pure measurement noise reaches on n points (97.5% of the
    time), or None. The noise level is the smallest training error of the data-only
    models (they assume no equation, so a wrong equation cannot raise it; a training
    error is at or below the noise, so this errs toward fewer ties).

    Why: with a few held-out points, a data-only model can score well below the noise
    by luck while a model that is right to within the noise scores at the noise, and
    the standard-error test cannot see that both are at the noise floor.

    cap: an RMS error the noise level may not exceed, used wherever a model of yours is
    involved (Result.tied). When the built-in data-only models all underfit, their
    training error is their misfit, not the noise, and without a cap models far worse
    than the lowest would tie with it. A built-in model compared with a model of yours
    that is lowest gets the generous cap, an upper confidence bound on the noise from
    the lowest model's held-out residuals (_noise_bound), because on a few held-out
    points a model can score well below the noise by luck; a model of yours compared
    with the lowest gets the strict cap, the lowest model's held-out RMS error itself.
    Both lean toward returning a built-in model. Built-in models compared with a
    built-in lowest keep the uncapped rule that stages A to I tested."""
    from scipy.stats import chi2
    # built-in data-only models only: a model of yours that nearly interpolates the
    # training points would pull the estimate toward zero
    s = [c.train_rmse for c in candidates
         if c.ok and not c.uses_equation and not c.yours and np.isfinite(c.train_rmse)]
    if not s or n < 1:
        return None
    level = min(s) if cap is None else min(min(s), cap)
    return float(chi2.ppf(NOISE_QUANTILE, n) / n * level ** 2)


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
        if e is None:
            return r[:1] + [c for c in r[1:] if _within_noise(c, r[0])]
        cands, n = self.candidates.values(), len(e)
        # Leans toward the built-in models both ways (README, "Your own model"):
        strict = _noise_floor(cands, n, float(np.sqrt(np.mean(e ** 2))))   # yours vs the lowest
        generous = _noise_floor(cands, n, _noise_bound(e))                 # built-ins vs yours lowest
        plain = _noise_floor(cands, n)                                     # built-ins vs a built-in

        def at_floor(c):
            f = strict if c.yours else generous if r[0].yours else plain
            return f is not None and c.holdout_errors is not None and \
                float(np.mean(c.holdout_errors ** 2)) <= f
        return r[:1] + [c for c in r[1:] if _within_noise(c, r[0]) or at_floor(c)]

    @property
    def best(self) -> Candidate:
        """pick="simplest": the simplest model among those tied with the lowest error.
        pick="lowest": the lowest error."""
        if not self.ranked:
            raise RuntimeError("no model finished; see result.table()")
        if self.pick == "lowest":
            return self.ranked[0]
        return min(self.tied, key=_simplicity)

    def __getitem__(self, name: str) -> Candidate:
        return self.candidates[name]

    def _constant_checks(self):
        """Per constant: (data-based problems, model disagreement or None, classical value,
        classical standard error)."""
        # yours are left out: their constants need not mean the equation's (a 'c' in
        # exp(-c*t) is half the c in u_tt + c*u_t + k*u = 0)
        good = [c for c in self.tied if c.uses_equation and c.params and not c.yours]
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
                elif self.best is self.ranked[0] and self.best.yours:
                    lines.append("      Yours has the lowest error and no built-in model is tied with it, "
                                 "so it is returned.")
                elif self.best is self.ranked[0] and any(c.yours for c in self.tied):
                    lines.append("      The lowest is a built-in model and first in the tie order, so it "
                                 "is returned.")
                elif self.best is self.ranked[0]:
                    lines.append("      The lowest is also the one the equation constrains most.")
                elif self.ranked[0].yours and not self.best.yours:
                    lines.append(f"      Yours has the lowest error, but on a tie a built-in model is "
                                 f"returned (equationnet cannot see how constrained yours is); "
                                 f"pick=\"lowest\" would return {self.ranked[0].title}.")
                elif self.best.yours:
                    lines.append(f"      Of your tied models, one you marked as using the equation is "
                                 f"returned; pick=\"lowest\" would return {self.ranked[0].title}.")
                elif any(c.yours for c in self.tied):
                    lines.append(f"      The first built-in model in the tie order is returned; "
                                 f"pick=\"lowest\" would return {self.ranked[0].title}.")
                else:
                    lines.append(f"      The one the equation constrains most is returned; pick=\"lowest\" "
                                 f"would return {self.ranked[0].title}.")
        if tied:
            pooled = lambda c: float(np.sqrt(np.mean(c.holdout_errors ** 2)))   # what ties use
            built_tied = [c for c in self.ranked if c.name in tied and not c.yours]
            # below a tied built-in model both in the table's error and in the pooled one
            skipped = [c for c in self.ranked if c.yours and c.name not in tied and built_tied
                       and pooled(c) < max(map(pooled, built_tied))
                       and c.holdout_rmse < max(b.holdout_rmse for b in built_tied)]
            if skipped:
                why = ("a more generous one, so that a tie goes to a built-in model" if self.ranked[0].yours
                       else "the data-only models' training error, which is far too high when those "
                            "fit badly")
                lines.append(f"      {', '.join(c.title for c in skipped)}: not marked tied, though "
                             "built-in models with larger errors are. Your models are judged against "
                             f"a noise level capped at the lowest error; built-in ones against {why}.")
        lines += [f"note ({c.title}): {c.note}" for c in order if c.ok and c.note]
        lines += [f"note: {n}" for n in self.notes]
        mine = [c for c in self.candidates.values() if c.yours]
        if mine:
            lines.append(f"yours: {', '.join(c.name for c in mine)}, scored on the same held-out points "
                         "as the built-in models. Constants are shown only if your model has "
                         "fitted_params(), standard errors only if it has uncertainty(). Saved "
                         "scripts for yours are stubs: equationnet cannot write out code it did "
                         "not write.")
        if any(c.param_se for c in self.ranked if not c.yours):
            lines.append("+-: one standard error, if the equation is exactly right (a wrong or "
                         "approximate equation can be off by more).")
        if any(c.param_se for c in self.ranked if c.yours):
            lines.append("+- on your models: as your model's uncertainty() reports it.")
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


# ---- your own models (YourModel) --------------------------------------------

def _is_model(obj) -> bool:
    return isinstance(obj, YourModel) or callable(getattr(obj, "predict", None)) or callable(obj)


def _model_items(models) -> list:
    if isinstance(models, (str, YourModel)):
        return [models]
    if isinstance(models, (list, tuple)):
        return list(models)
    if _is_model(models):
        return [models]
    return list(models)


def _your_models(items: list, taken: set) -> list:
    """Check the user's models; returns [(name, YourModel, trains)]. trains is False for
    a plain function, which is used as it is."""
    out = []
    for i, m in enumerate(items, 1):
        ym = m if isinstance(m, YourModel) else YourModel(m)
        obj = ym.model
        if isinstance(obj, type):
            raise ValueError(f"models: pass an instance of your model, e.g. {obj.__name__}(), "
                             f"not the class {obj.__name__}")
        trains = callable(getattr(obj, "fit", None)) and callable(getattr(obj, "predict", None))
        if not trains and not callable(obj):
            raise ValueError(f"models: {type(obj).__name__} is not a model. Pass a built-in model's "
                             f"name ({', '.join(MODELS)}), an object with fit(X, y) and predict(X) "
                             "methods, or a function predict(X)")
        name = ym.name if ym.name is not None else (
            type(obj).__name__ if trains else getattr(obj, "__name__", ""))
        name = str(name)
        if ym.name is None and name in ("", "<lambda>", "function"):
            name = f"your_model_{i}"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,39}", name):
            bad = sorted({ch for ch in name if not (ch.isascii() and (ch.isalnum() or ch in "_-"))})
            why = (f" (it contains {', '.join(map(repr, bad))})" if bad else
                   " (it is too long)" if len(name) > 40 else " (it must start with a letter or '_')")
            raise ValueError(f"models: the name {name!r} cannot be used{why}; use letters, digits, "
                             "'_' or '-', starting with a letter or '_', at most 40 characters "
                             "(en.custom(model, name=...))")
        if name in MODELS:
            raise ValueError(f"models: {name!r} is the name of a built-in model; give yours another "
                             "name with en.custom(model, name=...)")
        if name in taken:
            raise ValueError(f"models: two models are called {name!r}; give yours another name with "
                             "en.custom(model, name=...)")
        taken.add(name)
        out.append((name, ym, trains))
    return out


def _predictions(p, n: int, what: str) -> np.ndarray:
    a = np.asarray(p, float)
    if a.ndim > 1 and a.size == n:
        a = a.reshape(-1)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{what} returned an array of shape {np.shape(p)} for {n} points; "
                         f"it must return {n} values, one per row of X")
    return a


def _your_namespace(trains: bool) -> dict:
    if trains:
        def predict(state, X):
            return _predictions(state.predict(X), len(X), "predict(X)")
    else:
        def predict(state, X):
            return _predictions(state(X), len(X), "your function")

    def fitted_params(state):
        f = getattr(state, "fitted_params", None)
        return {str(k): float(v) for k, v in f().items()} if callable(f) else {}

    def uncertainty(state):
        f = getattr(state, "uncertainty", None)
        return {str(k): float(v) for k, v in f().items()} if callable(f) else {}

    return {"predict": predict, "fitted_params": fitted_params, "uncertainty": uncertainty}


def _train_eval_yours(ym: YourModel, trains: bool, X, y, train, test):
    """Train a fresh copy of the user's model on `train`; same returns as _train_eval."""
    ns = _your_namespace(trains)
    if trains:
        try:
            state = copy.deepcopy(ym.model)
        except Exception as exc:
            raise TypeError(f"your model could not be copied ({type(exc).__name__}: {exc}); "
                            "equationnet trains a fresh copy for every held-out split") from None
        state.fit(X[train], y[train])
    else:
        state = ym.model

    def rmse(r):
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")

    r_test = ns["predict"](state, X[test]) - y[test] if test.any() else np.zeros(0)
    r_train = ns["predict"](state, X[train]) - y[train]
    return ns, state, rmse(r_train), rmse(r_test), r_test


def _stub(name: str, ym: YourModel, trains: bool, indep: list, target: str) -> str:
    """The saved 'script' for a model of yours: how it was scored, not the model itself."""
    obj = ym.model
    src = type(obj) if trains else obj
    mod, qual = getattr(src, "__module__", None), getattr(src, "__qualname__", "")
    top = qual.split(".")[0]
    if mod and mod != "__main__" and top and "<" not in qual:
        # sklearn.ensemble._gb -> sklearn.ensemble: private submodules are not where users import from
        parts = mod.split(".")
        while len(parts) > 1 and parts[-1].startswith("_"):
            parts.pop()
        imp = f"# from {'.'.join(parts)} import {top}"
    else:
        imp = "# (defined in your own script or notebook: import it or paste it here)"
    args = "..."
    get_params = getattr(obj, "get_params", None)
    if trains and callable(get_params):
        try:
            args = ", ".join(f"{k}={v!r}" for k, v in get_params(deep=False).items())
        except Exception:
            pass
    how = ([f"# model = {qual}({args})",
            "# model.fit(X_train, y_train)     # with cv > 1, the returned model: all your data",
            "# predictions = model.predict(X)"] if trains else
           [f"# predictions = {qual or name}(X)   # used as it is; equationnet did not train it"])
    return "\n".join([
        f'"""{name}: your own model, as equationnet scored it.',
        "",
        "This is a stub, not a standalone script: equationnet cannot write out code it did not",
        "write. It records the model's class, its parameters where the model reports them",
        "(get_params()), and how it was trained and called. It does not record which points",
        "were held out: run en.fit again with the same data, split, holdout, cv and seed to",
        "get the same split.",
        '"""',
        imp,
        "",
        f"# X: one row per point, columns {', '.join(indep)} (in this order); y: {target}",
        *how,
        "",
    ])


def fit(equation: str, data, target: str = "u", known: dict | None = None,
        models: list[str] | str = "all", split: str = "random", holdout: float = 0.2,
        seed: int = 0, cv: int | str = 1, predict_at=None, config: dict | None = None,
        pick: str = "simplest", verbose: bool = True) -> Result:
    """Train every applicable model and rank them by error on held-out data.

    equation   : text, e.g. "u_tt + c*u_t + k*u = 0"
    data       : CSV path, dict of columns, or DataFrame; one column per variable plus the target
    known      : values for constants you already know, e.g. {"k": 4.0}
    models     : "all" (default), built-in model names, and your own models: anything with
                 fit(X, y) and predict(X), or a function predict(X); wrap one in
                 custom(model, name=..., uses_equation=...) to name it. Yours are scored
                 on the same held-out points, e.g. models=["all", my_model].
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
    names, others = [], []
    for m in _model_items(models):
        if isinstance(m, str):
            for n in (list(MODELS) if m == "all" else [m]):
                if n not in names:
                    names.append(n)
        else:
            others.append(m)
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {', '.join(map(repr, unknown))}; models: {', '.join(MODELS)}")
    yours = _your_models(others, set())
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
    for name, ym, trains in yours:
        title = f"{name} (yours{', uses equation' if ym.uses_equation else ''})"
        cand = Candidate(name, title, ym.uses_equation, yours=True,
                         script=_stub(name, ym, trains, indep, target))
        out[name] = cand
        t0 = time.perf_counter()
        errors = []
        try:
            for i, (tr, te) in enumerate(folds):
                try:
                    ns, state, rm_tr, rm_te, r_te = _train_eval_yours(ym, trains, X, y, tr, te)
                except Exception as exc:
                    if k > 1:
                        raise RuntimeError(f"in fold {i + 1} of {k}: {type(exc).__name__}: {exc}") from exc
                    raise
                cand.fold_rmse.append(rm_te)
                errors.append(r_te)
            if k > 1:
                ns, state, rm_tr, _, _ = _train_eval_yours(ym, trains, X, y, final, ~final)
            cand.namespace, cand.state = ns, state
            cand.train_rmse = rm_tr
            cand.holdout_rmse = float(np.mean(cand.fold_rmse))
            cand.fold_errors = errors
            cand.params = ns["fitted_params"](state)
            cand.param_se = ns["uncertainty"](state)
            if not np.isfinite(cand.holdout_rmse):
                cand.status = "failed: your model predicts NaN or infinite values on held-out points"
        except Exception as exc:
            cand.status = f"failed: {type(exc).__name__}: {exc}"
            if verbose > 1:
                traceback.print_exc()
        cand.seconds = time.perf_counter() - t0
        if verbose:
            msg = f"{cand.holdout_rmse:.4g}" if cand.ok else cand.status
            print(f"  {title:<34} {msg}  ({cand.seconds:.1f}s)")
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
