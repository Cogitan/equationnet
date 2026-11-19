"""Script templates. Each renders to a complete, standalone Python file.

Every script defines the same four functions, which is how equationnet runs it:
    train(X, y, seed=0) -> state
    predict(state, X) -> array
    fitted_params(state) -> dict of fitted constants
    learned_function(state, name, z) -> values of an unknown function (if any)
"""

HEADER = '''"""
equationnet <<VERSION>>  |  model: <<MODEL>>
<<MODEL_DOC>>

Equation : <<EQUATION>>
Fits     : <<FITS>>
Data     : CSV with a header row and columns <<COLUMNS>>
Run      : python this_file.py data.csv [points.csv]   (points: where to predict, e.g. future times)
Needs    : <<NEEDS>>
"""
import sys

import numpy as np

INDEP = <<INDEP>>      # independent variables (CSV columns, in this order)
TARGET = "<<TARGET>>"
PARAMS = <<PARAMS>>    # unknown constants this script fits
UFUNCS = <<UFUNCS>>    # unknown functions this script learns
'''

FOOTER = '''

def load_csv(path):
    """Read the columns this script needs from a comma-separated file with a header row.
    Other columns are ignored (and named); rows with a missing value are left out (and
    their numbers printed); anything that is not a number stops with a message."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:   # -sig: Excel's byte-order mark
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows:
        sys.exit(f"{path} is empty")
    names = [h.strip() for h in rows[0]]
    need = INDEP + [TARGET]
    missing = [n for n in need if n not in names]
    if missing:
        sys.exit(f"{path} has no column {', '.join(repr(m) for m in missing)} (columns: {', '.join(names)}); "
                 f"this script needs {', '.join(need)}")
    extra = [n for n in names if n not in need]
    if extra:
        print(f"ignored column(s) not used by this script: {', '.join(extra)}")
    idx = [names.index(n) for n in need]
    values, left_out = [], []
    for i, r in enumerate(rows[1:], start=1):
        if len(r) != len(names):
            sys.exit(f"data row {i} has {len(r)} values; the header has {len(names)}")
        row = []
        for j in idx:
            cell = r[j].strip()
            if cell.lower() in ("", "nan", "na", "n/a", "null", "none", "-"):
                row = None
                break
            try:
                row.append(float(cell))
            except ValueError:
                sys.exit(f"data row {i}, column '{names[j]}': {cell!r} is not a number")
        if row is None or not np.all(np.isfinite(row)):
            left_out.append(i)
        else:
            values.append(row)
    if left_out:
        print(f"left out {len(left_out)} row(s) with a missing or non-finite value: "
              f"data rows {', '.join(map(str, left_out[:10]))}{', ...' if len(left_out) > 10 else ''}")
    if len(values) < 5:
        sys.exit(f"need at least 5 data rows with every value present; got {len(values)}")
    data = np.asarray(values, float)
    return data[:, :-1], data[:, -1]


def load_points(path):
    """Locations to predict at: a CSV with the variable columns (others are ignored, and named)."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows:
        sys.exit(f"{path} is empty")
    names = [h.strip() for h in rows[0]]
    missing = [n for n in INDEP if n not in names]
    if missing:
        sys.exit(f"{path} has no column {', '.join(repr(m) for m in missing)}; "
                 f"points to predict at need {', '.join(INDEP)}")
    extra = [n for n in names if n not in INDEP]
    if extra:
        print(f"ignored column(s) in {path}: {', '.join(extra)} (only {', '.join(INDEP)} are used)")
    idx = [names.index(n) for n in INDEP]
    pts = []
    for i, r in enumerate(rows[1:], start=1):
        try:
            pts.append([float(r[j]) for j in idx])
        except (ValueError, IndexError):
            sys.exit(f"{path}, data row {i}: the variable values must be numbers")
    return np.asarray(pts, float)


if __name__ == "__main__":
    import os

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Usage: python this_file.py data.csv [points.csv]")
        print("  data.csv    training data; predictions at its rows go to data_<script>_pred.csv")
        print("  points.csv  optional: other locations (e.g. future times) to predict at")
        sys.exit(0 if len(sys.argv) >= 2 else 1)
    name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "<<MODEL>>"
    X, y = load_csv(sys.argv[1])
    state = train(X, y)
    pred = predict(state, X)
    se = uncertainty(state) if "uncertainty" in globals() else {}
    print("fitted:", {k: round(v, 6) for k, v in fitted_params(state).items()})
    if se:
        print("standard error:", {k: round(v, 6) for k, v in se.items()})
    print("rms error on the data:", float(np.sqrt(np.mean((pred - y) ** 2))))
    out = os.path.splitext(sys.argv[1])[0] + f"_{name}_pred.csv"
    np.savetxt(out, np.column_stack([X, y, pred]), delimiter=",",
               header=",".join(INDEP + [TARGET, "prediction"]), comments="")
    print("predictions at the data written to", out)
    if len(sys.argv) > 2:
        P = load_points(sys.argv[2])
        out2 = os.path.splitext(sys.argv[2])[0] + f"_{name}_pred.csv"
        np.savetxt(out2, np.column_stack([P, predict(state, P)]), delimiter=",",
                   header=",".join(INDEP + ["prediction"]), comments="")
        print("predictions at", sys.argv[2], "written to", out2)
'''

# --------------------------------------------------------------------------
# Classical: solve the equation numerically, fit constants by least squares.
# --------------------------------------------------------------------------

UFUNC_POLY = '''
UDEG = 3  # unknown functions are fitted as cubic polynomials of their argument


def _poly(coef):
    return lambda z: np.polynomial.polynomial.polyval(z, coef)


def standard_errors(J, r):
    """Standard error of every fitted number, from the fit's Jacobian J (residuals r):
    covariance = s^2 (J^T J)^-1 with s^2 the residual variance. Uses the SVD, so a
    direction the data do not constrain at all gives an infinite error instead of being
    dropped. A linear approximation: good when the error is small against the value."""
    dof = len(r) - J.shape[1]
    if dof <= 0 or not np.all(np.isfinite(J)):
        return np.full(J.shape[1], np.inf)
    s2 = float(np.sum(r ** 2)) / dof
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    if S[0] == 0:
        return np.full(J.shape[1], np.inf)
    flat = S < 1e-10 * S[0]
    w = np.where(flat, 0.0, 1 / np.where(flat, 1.0, S) ** 2)
    var = s2 * (Vt.T ** 2 @ w)
    blind = (Vt.T ** 2 @ flat.astype(float)) > 1e-6       # touches an unconstrained direction
    return np.where(blind, np.inf, np.sqrt(var))
'''

EQUATION_FIT_ODE = '''
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

ORDER = <<ORDER>>  # highest derivative of the target in the equation
MAX_WORK = 5000    # solver evaluations allowed per solve (at least 50 per data point)
''' + UFUNC_POLY + '''

<<EXPLICIT_FN>>


def _split(theta):
    k = len(PARAMS)
    P = dict(zip(PARAMS, theta[:k]))
    ic = theta[k:k + ORDER]
    k += ORDER
    F = {}
    for name in UFUNCS:
        F[name] = _poly(theta[k:k + UDEG + 1])
        k += UDEG + 1
    return P, ic, F


class _TooCostly(Exception):
    pass


def simulate(theta, t0, t, limit=np.inf):
    """Solve the equation from t0 (forwards and backwards) and return u at times t.
    Returns None if the solver fails, u runs past `limit` (a runaway guess), or the
    solve needs more than MAX_WORK evaluations: a guess that oscillates thousands of
    times across the data (e.g. constants of order 1 when time is in milliseconds)
    would otherwise take minutes for one solve."""
    P, ic, F = _split(theta)
    calls = [0]
    budget = max(MAX_WORK, 50 * len(t))

    def rhs(tt, s):
        calls[0] += 1
        if calls[0] > budget:
            raise _TooCostly
        return [*s[1:], highest(tt, *s, P, F)]

    def runaway(tt, s):
        return limit - abs(s[0])

    runaway.terminal = True
    out = np.empty(len(t))
    for mask in (t >= t0, t < t0):
        if not mask.any():
            continue
        ts = np.unique(t[mask])
        forward = ts[0] >= t0
        end = ts.max() if forward else ts.min()
        if end == t0:
            out[mask] = ic[0]
            continue
        try:
            with np.errstate(all="ignore"):
                sol = solve_ivp(rhs, (t0, end), ic, t_eval=ts if forward else ts[::-1],
                                method="LSODA", rtol=1e-8, atol=1e-10, events=runaway)
        except _TooCostly:
            return None
        if not sol.success or sol.status == 1 or sol.y.shape[1] != len(ts):
            return None
        vals = sol.y[0] if forward else sol.y[0][::-1]
        out[mask] = np.interp(t[mask], ts, vals)
    return out


def _smooth(t, y):
    """A smooth polynomial through the data, used only for starting guesses."""
    return np.polynomial.Chebyshev.fit(t, y, int(min(12, max(2, len(t) // 3))))


def _regression_start(t, y, cheb):
    """Starting constants from the data's own derivatives: fit the equation to the
    smooth curve's u, u', u'' directly (fast, no solver). Returns None on failure."""
    k = len(PARAMS) + len(UFUNCS) * (UDEG + 1)
    if k == 0:
        return np.zeros(0)
    tt = np.linspace(t.min(), t.max(), 200)
    states = [cheb.deriv(m)(tt) if m else cheb(tt) for m in range(ORDER)]
    top = cheb.deriv(ORDER)(tt)

    def r(v):
        P = dict(zip(PARAMS, v[:len(PARAMS)]))
        F, j = {}, len(PARAMS)
        for name in UFUNCS:
            F[name] = _poly(v[j:j + UDEG + 1])
            j += UDEG + 1
        with np.errstate(all="ignore"):
            out = np.broadcast_to(highest(tt, *states, P, F), tt.shape) - top
        return np.where(np.isfinite(out), out, 1e6)

    # Constants can be any size and either sign: screen 24 decades (1e-12 to 1e12, so
    # times in microseconds or in years both work), then zoom twice around the best
    # (unknown functions start at zero), and refine from the best.
    n = len(PARAMS)
    v0 = np.concatenate([np.ones(n), np.zeros(k - n)])
    if n:
        rng = np.random.default_rng(0)
        F0 = {name: _poly(np.zeros(UDEG + 1)) for name in UFUNCS}

        def loss(vals):
            P = {p: vals[:, i:i + 1] for i, p in enumerate(PARAMS)}
            with np.errstate(all="ignore"):
                out = np.broadcast_to(highest(tt, *states, P, F0), (len(vals), len(tt))) - top
                v = np.mean(out ** 2, axis=1)
            return np.where(np.isfinite(v), v, np.inf)

        def draw(center, half, m):
            return (np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)
                    * 10 ** (center + half * (2 * rng.random((m, n)) - 1)))

        cand = np.vstack([np.ones((1, n)), draw(np.zeros(n), 12.0, 16384)])
        best = cand[np.argmin(loss(cand))]
        for half in (1.5, 0.3):
            cand = np.vstack([best, np.sign(best) * np.abs(draw(np.log10(np.abs(best)), half, 4096))])
            best = cand[np.argmin(loss(cand))]
        v0[:n] = best
    try:
        return least_squares(r, v0, x_scale="jac").x
    except Exception:
        return None


def _state_at(theta, t_from, t_to):
    """u and its derivatives at t_to, solved from the start values theta holds for
    t_from; None if the solve fails or is too costly."""
    P, ic, F = _split(theta)
    calls = [0]

    def rhs(tt, s):
        calls[0] += 1
        if calls[0] > MAX_WORK:
            raise _TooCostly
        return [*s[1:], highest(tt, *s, P, F)]

    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(rhs, (t_from, t_to), ic, method="LSODA", rtol=1e-8, atol=1e-10)
    except _TooCostly:
        return None
    if not sol.success:
        return None
    out = sol.y[:, -1]
    return out if np.all(np.isfinite(out)) else None


def _start_earlier(best, t, y, t0, t_want, scale, limit, loc_at):
    """Move the fitted solution's start from the first training point t0 back to t_want,
    where predictions are first wanted, so predictions there run forwards: run back, a
    damped equation multiplies small errors without bound (a held-out first point once
    scored 78,000 against 0.05). When this returns None the start stays at t0 and the
    caller's note says that predictions before it are solved backwards.

    The new start's first guess is the fitted solution run back to t_want when that is
    stable (differences grow less than STABLE times over the gap). Otherwise the data
    may say nothing about the earlier start (an equation that forgets its start quickly),
    and the guess is a small polynomial through the first points, extrapolated. A weak
    pull toward that guess (START_PULL) decides only what the data leave open; it is
    weak because the guess can be far off, and a pull the strength of one measurement
    moved a well-determined pendulum into the wrong valley. Then one refit of
    everything. Returns (state, note or None), or None to keep the fit as it was."""
    n = len(PARAMS)
    theta = best.x
    back = _state_at(theta, t0, t_want)
    stable = False
    if back is not None:
        grow = 0.0
        for i in range(ORDER):
            h = 1e-6 * (abs(theta[n + i]) + scale)
            pert = theta.copy()
            pert[n + i] += h
            b = _state_at(pert, t0, t_want)
            if b is None:
                grow = np.inf
                break
            grow = max(grow, float(np.max(np.abs(b - back))) / h)
        stable = grow <= STABLE and np.all(np.abs(back) <= limit)
    if stable:
        guess, dt = back, max(t0 - t_want, 1e-300)
        source = "the fitted solution run back from the first training point"
    elif loc_at is not None:
        guess, dt = loc_at
        source = "a guess read off the first few points"
    else:
        return None
    w = START_PULL * dt ** np.arange(ORDER) / scale

    def safe(th):
        sim = simulate(th, t_want, t, limit)
        r = np.full(len(t), 1e3) if sim is None or not np.all(np.isfinite(sim)) else (sim - y) / scale
        return np.concatenate([r, w * (th[n:n + ORDER] - guess)])

    try:
        f = least_squares(safe, np.concatenate([theta[:n], guess, theta[n + ORDER:]]),
                          x_scale="jac", max_nfev=300, diff_step=DIFF_STEP)
    except Exception:
        return None
    m = len(t)
    data_cost = float(np.sum(f.fun[:m] ** 2))
    if not data_cost <= 1.5 * 2 * best.cost + 1e-12:   # the refit must fit the data as well
        return None
    # standard errors: the pull rows stay in the Jacobian (they keep an undetermined start
    # finite) but are not data, so the residual variance uses the data alone and the
    # data's degrees of freedom (counting the pull rows made every error too small by
    # sqrt((m - p) / (m - p + ORDER))).
    p_all = f.jac.shape[1]
    if m <= p_all:
        return None
    r_eq = np.concatenate([f.fun[:m] * np.sqrt((m + ORDER - p_all) / (m - p_all)), np.zeros(ORDER)])
    se = standard_errors(f.jac, r_eq)[:n]
    state = {"theta": f.x, "t0": t_want, "se": dict(zip(PARAMS, map(float, se)))}
    # can the data alone place u(t_want) within half the data's spread?
    se_start = standard_errors(f.jac[:m], f.fun[:m])[n]
    note = None
    if not se_start <= 0.5 * scale:
        note = (f"u at t = {t_want:.4g}, where predictions are wanted, is not determined by the "
                f"data: the equation forgets where it started before the first training point "
                f"(t = {t0:.4g}). Before that point the model follows {source}.")
    return state, note


SCREEN = 128   # random guesses scored by one cheap solve each (0 = none)
STARTS = 6   # best-scoring distinct guesses that get a full fit
DIFF_STEP = 1e-4  # relative step for the fit's finite-difference gradients (None = SciPy's)
DOMAIN = None     # where predictions are wanted: [[low], [high]] in time; None = the data's range.
PROFILE = 0.5     # profile check: move each constant by this fraction of its value (0 = off)
PROFILE_CHI2 = 3.84  # ... and call it loose if the refit is within this (95%, one number)
WINDOWS = (0.25, 0.5)   # second-order and higher: also search on the first quarter of the
                        # time range, then extend to half, then all (0 entries = off)
STABLE = 10.0     # running the fit back to an earlier start counts as stable while differences
                  # grow less than this many times over the gap
START_PULL = 1e-3 # weight of the pull toward the guessed earlier start (see _start_earlier)


def train(X, y, seed=0, domain=None):
    t = X[:, 0]
    t0 = float(t.min())
    domain = domain if domain is not None else DOMAIN
    t_want = t0 if domain is None else min(t0, float(np.ravel(domain[0])[0]))
    scale = float(np.std(y)) or 1.0
    limit = 100 * (np.max(np.abs(y)) + scale)
    cheb = _smooth(t, y)
    ic0 = np.array([float(cheb.deriv(m)(t0)) if m else float(cheb(t0)) for m in range(ORDER)])
    nu = len(UFUNCS) * (UDEG + 1)
    rng = np.random.default_rng(seed)
    n = len(PARAMS)

    ALL = np.ones(len(t), bool)

    def resid(theta, m=ALL):
        sim = simulate(theta, t0, t[m], limit)
        if sim is None or not np.all(np.isfinite(sim)):
            return None
        return (sim - y[m]) / scale

    def safe(theta, m=ALL):
        r = resid(theta, m)
        return np.full(int(m.sum()), 1e3) if r is None else r

    # Starting values to try for u(t0) and its derivatives: the smooth curve's, and a
    # small polynomial's through the first few points (the smooth curve can be far off
    # at the ends when the data swing many times, and a wrong starting slope sends an
    # oscillator fit into the wrong valley); both are tried and the better kept. Then the
    # first data point's value, and points across the data's range (a noisy smooth curve
    # can start where the equation runs away, e.g. below zero for growth).
    o = np.argsort(t)[:min(len(t), 2 * ORDER + 3)]
    try:
        loc = np.polynomial.Polynomial.fit(t[o], y[o], min(ORDER + 1, len(o) - 1))
        ic_loc = np.array([float(loc.deriv(m)(t0)) if m else float(loc(t0)) for m in range(ORDER)])
    except Exception:
        ic_loc = ic0
    first = float(y[np.argmin(t)])
    ic_options = [ic0, ic_loc] + [np.concatenate([[v], ic0[1:]]) for v in
                                  [first, *np.quantile(y, [0.05, 0.25, 0.5, 0.75, 0.95])]]

    def with_start(g, all_options=False, m=ALL):
        """Full parameter vector for guess g, with the starting values (u and its
        derivatives at t0) refitted by one Gauss-Newton step: exact when the equation is
        linear in u. The better of the first two starting values is used, else the first
        of the others that solves (all of them, keeping the best, when all_options).
        Returns (theta, cost); cost is inf if none solves."""
        if all_options:
            return min((with_start_from(g, ic, m) for ic in ic_options), key=lambda s: s[1])
        out = min((with_start_from(g, ic, m) for ic in ic_options[:2]), key=lambda s: s[1])
        if np.isfinite(out[1]):
            return out
        for ic in ic_options[2:]:
            out = with_start_from(g, ic, m)
            if np.isfinite(out[1]):
                return out
        return out

    def with_start_from(g, ic, m=ALL):
        theta = np.concatenate([g[:n], ic, g[n:]])
        r0 = resid(theta, m)
        if r0 is None:
            return theta, np.inf
        h = 1e-3 * scale
        cols = []
        for i in range(n, n + ORDER):
            r = resid(theta + h * np.eye(len(theta))[i], m)
            if r is None:
                return theta, float(np.sum(r0 ** 2))
            cols.append((r - r0) / h)
        trial = theta.copy()
        trial[n:n + ORDER] += np.linalg.lstsq(np.column_stack(cols), -r0, rcond=None)[0]
        r1 = resid(trial, m)
        c0, c1 = float(np.sum(r0 ** 2)), np.inf if r1 is None else float(np.sum(r1 ** 2))
        return (trial, c1) if c1 < c0 else (theta, c0)

    guesses = []
    reg = _regression_start(t, y, cheb)
    if reg is not None:
        guesses.append(reg)
    for v in (1.0, 0.1, 10.0):
        guesses.append(np.concatenate([v * np.ones(n), np.zeros(nu)]))
    for _ in range(SCREEN if n + nu else 0):
        g = np.concatenate([10 ** rng.uniform(-2, 2, n), np.zeros(nu)])
        for j in range(len(UFUNCS)):   # unknown functions start as a random straight line
            g[n + j * (UDEG + 1) + 1] = rng.choice([-1.0, 1.0]) * 10 ** rng.uniform(-2, 2)
        guesses.append(g)
    if n == 0 and nu == 0:
        guesses = [np.zeros(0)]

    def search(m=ALL):
        """Score every guess with one solve, then fully fit the best few that differ."""
        scored = sorted((with_start(g, all_options=(reg is not None and g is reg), m=m) + (g,)
                         for g in guesses), key=lambda s: s[1])
        chosen, keys = [], []
        for theta0, cost, g in scored:
            key = np.sign(g) * np.log10(np.abs(g) + 1e-3)
            if any(np.max(np.abs(key - k)) < 0.3 for k in keys):
                continue
            chosen.append(theta0)
            keys.append(key)
            if len(chosen) == STARTS:
                break
        out = []
        for theta0 in chosen:
            try:
                out.append(least_squares(lambda th: safe(th, m), theta0, x_scale="jac",
                                         max_nfev=300, diff_step=DIFF_STEP))
            except Exception:
                continue
        return out

    fits = search()

    # Second- and higher-order equations (oscillators) also get a search on the first
    # part of the time range only (about one swing), extended to the whole range one step
    # at a time. Fitted over many swings, an oscillator has a valley for every wrong
    # frequency and the whole-range search can miss the right one (Duffing, uneven data:
    # every fit 25 times the noise). Over one swing the wrong valleys are far away. These
    # fits join the others and the choice below is unchanged: the lowest-cost fit wins,
    # unless others tie with it within the noise, when the smoothest of those is kept.
    if ORDER >= 2 and WINDOWS and n + nu:
        theta = None
        for frac in WINDOWS:
            m = t <= t0 + frac * (t.max() - t0)
            if m.sum() < 2 * (n + ORDER + nu) + 2:
                continue
            try:
                if theta is None:
                    wf = search(m)
                    theta = min(wf, key=lambda f: f.cost).x if wf else None
                else:
                    theta = least_squares(lambda th: safe(th, m), theta, x_scale="jac",
                                          max_nfev=300, diff_step=DIFF_STEP).x
            except Exception:
                theta = None
                break
        if theta is not None:
            try:
                fits.append(least_squares(safe, theta, x_scale="jac", max_nfev=300, diff_step=DIFF_STEP))
            except Exception:
                pass
    if not fits:
        raise RuntimeError("no starting guess converged")

    # Fits that tie within the noise are indistinguishable by the data (on evenly spaced
    # samples a fast oscillation can pass through exactly the same points as a slow one).
    # Among those, keep the smoothest curve: the fewest turning points between the data.
    low = min(f.cost for f in fits)
    noise = 2 * low / max(1, len(t) - len(fits[0].x))          # residual variance estimate
    dense = np.linspace(t.min(), t.max(), 20 * len(t))

    def turns(theta):
        sim = simulate(theta, t0, dense, limit)
        return np.inf if sim is None else int(np.sum(np.diff(np.sign(np.diff(sim))) != 0))

    tied = [f for f in fits if 2 * (f.cost - low) <= 4 * noise]
    best = min(tied, key=lambda f: (turns(f.x), f.cost))
    se = standard_errors(best.jac, best.fun)[:len(PARAMS)]
    state = {"theta": best.x, "t0": t0, "se": dict(zip(PARAMS, map(float, se)))}
    # Profile check: the standard errors treat the fit as a straight line near the best
    # constants, and a curved valley of equally good fits can make a constant look pinned
    # down when it is not (a second-order equation on a single exponential: k = -0.55 +-
    # 0.24 while k = 2.2 fits as well). Move each constant to half and to one and a half
    # times its value, refit the rest, and call it loose if the fit is still as good
    # (within the 95% chi-squared limit for one number).
    dof = max(1, len(t) - len(best.x))
    s2 = 2 * best.cost / dof
    loose = []
    for i, name in enumerate(PARAMS):
        v = best.x[i]
        if PROFILE <= 0 or v == 0 or not np.isfinite(v) or s2 <= 0:
            continue
        Ji, Jr = best.jac[:, i], np.delete(best.jac, i, axis=1)
        for val in (v * (1 - PROFILE), v * (1 + PROFILE)):
            rest = np.delete(best.x, i)
            # two starts for the others: as they are, and moved along the valley's local
            # direction (the fit's own linear prediction of how they follow this constant)
            starts = [rest]
            try:
                starts.append(rest - np.linalg.lstsq(Jr, Ji * (val - v), rcond=None)[0])
            except Exception:
                pass

            def fixed(r, val=val, i=i):
                return safe(np.insert(r, i, val))

            cost = np.inf
            for r0 in starts:
                try:
                    cost = min(cost, least_squares(fixed, r0, x_scale="jac", max_nfev=100,
                                                   diff_step=DIFF_STEP).cost)
                except Exception:
                    continue
            if 2 * (cost - best.cost) / s2 <= PROFILE_CHI2:
                loose.append(name)
                break
    if loose:
        state["loose"] = loose
    notes = []
    if t_want < t0:
        try:
            loc_at = (np.array([float(loc.deriv(m)(t_want)) if m else float(loc(t_want))
                                for m in range(ORDER)]), max(float(np.ptp(t[o])), 1e-300))
        except Exception:
            loc_at = None
        moved = _start_earlier(best, t, y, t0, t_want, scale, limit, loc_at)
        if moved is not None:
            state, note = dict(moved[0], **({"loose": loose} if loose else {})), moved[1]
            if note:
                notes.append(note)
        else:
            notes.append(f"predictions before the first training point (t = {t0:.4g}) are solved "
                         "backwards from it, because the start could not be moved earlier; for "
                         "a damped equation that can multiply small errors, so treat them with care.")
    others = [turns(f.x) for f in tied if f is not best]
    mine = turns(best.x)
    faster = [k for k in others if np.isfinite(k) and k > 1.5 * mine + 1]
    if faster:
        notes.insert(0, f"a faster-changing curve ({int(max(faster))} turning points against "
                        f"{int(mine)}) fits these data equally well; the smoother one is "
                        "returned. More closely or unevenly spaced data would tell them apart.")
    if notes:
        state["note"] = " Also: ".join(notes)
    return state


def predict(state, X):
    sim = simulate(state["theta"], state["t0"], X[:, 0])
    return np.full(len(X), np.nan) if sim is None else sim


def fitted_params(state):
    return {p: float(v) for p, v in zip(PARAMS, state["theta"][:len(PARAMS)])}


def uncertainty(state):
    """Standard error of each fitted constant (inf = not determined by the data)."""
    return dict(state.get("se", {}))


def learned_function(state, name, z):
    return _split(state["theta"])[2][name](np.asarray(z, float))
'''

EQUATION_FIT_PDE = '''
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
from scipy.optimize import least_squares
from scipy.sparse import diags

XI, TI = <<XI>>, <<TI>>  # which columns are space and time
NX = 81                # grid points in space for the solver (fourth-order differences)
NX_MAX = 321           # the grid check refines up to this many points
GRID_TOL = 0.2         # refine while the constants move by more than this many standard errors
RTOL = 1e-7            # time-stepping tolerance of the solver
N_START = 25           # start profile = smoothed data + a fitted correction at up to this many inner x points
N_EDGE = 4             # each edge = smoothed data + a fitted correction at this many times
EDGE_ROUNDS = 3        # rounds of fitting the corrections, then the constants (0 = data as is)
JOINT_STEPS = 30       # then steps of fitting the constants and corrections together
DIFF_STEP = 1e-3       # finite-difference step for the fit: well above the solver's own error
SCREEN = 16            # guesses for the constants scored by one solve each; the best is fitted
''' + UFUNC_POLY + '''

<<EXPLICIT_FN>>


def _split(theta):
    k = len(PARAMS)
    P = dict(zip(PARAMS, theta[:k]))
    F = {}
    for name in UFUNCS:
        F[name] = _poly(theta[k:k + UDEG + 1])
        k += UDEG + 1
    return P, F


def _unpack(theta, E):
    """theta = [constants, unknown-function coefficients, corrections to the start
    profile, the left edge and the right edge]."""
    k = len(PARAMS) + len(UFUNCS) * (UDEG + 1)
    a, b = len(E["xn"]), len(E["tn"])
    P, F = _split(theta[:k])
    return P, F, theta[k:k + a], theta[k + a:k + a + b], theta[k + a + b:k + a + 2 * b]


def _sorted(a, b):
    """Sorted, de-noised samples of a 1D profile: a smoothing spline when there are
    enough points (so measurement noise is not fed into every solver step)."""
    a, inv = np.unique(a, return_inverse=True)
    b = np.bincount(inv, b) / np.bincount(inv)
    if len(a) >= 5:
        from scipy.interpolate import make_smoothing_spline
        grid = np.linspace(a[0], a[-1], 201)
        return grid, make_smoothing_spline(a, b)(grid)
    return a, b


def edges(X, y):
    """Start profile and edge values from the data, smoothed, plus where their fitted
    corrections live. Training fits the corrections along with the constants, so
    missing or noisy points at the start or the edges are repaired by the equation."""
    x, t = X[:, XI], X[:, TI]
    xs, ts = np.unique(x), np.unique(t)
    left, right = np.isclose(x, xs[0]), np.isclose(x, xs[-1])
    # first guess at the start profile: the earliest time slices that together cover
    # at least 60% of the x positions (the fitted correction then repairs it)
    for tj in ts:
        early = t <= tj
        if len(np.unique(x[early])) >= max(3, 0.6 * len(xs)):
            break
    if len(xs) < 3 or early.sum() < 3 or left.sum() < 2 or right.sum() < 2:
        raise ValueError("needs data on a grid: at least 3 points near the first time "
                         "and 2 on each edge in x")
    return {"t0": float(ts[0]), "xlo": float(xs[0]), "xhi": float(xs[-1]), "nx": NX,
            "xn": np.linspace(xs[0], xs[-1], min(len(xs), N_START + 2))[1:-1],
            "tn": np.linspace(ts[0], ts[-1], min(len(ts), N_EDGE)),
            "ic": _sorted(x[early], y[early]), "bl": _sorted(t[left], y[left]),
            "br": _sorted(t[right], y[right])}


def _correction(E, v, lo=0.0, hi=0.0):
    """Smooth correction to the start profile, fitted at the inner points xn; at the two
    x edges it is lo and hi, chosen by the caller so the start profile meets the edge
    values (each edge has its own fitted values)."""
    xs = np.concatenate([[E["xlo"]], E["xn"], [E["xhi"]]])
    vs = np.concatenate([[lo], v, [hi]])
    return make_interp_spline(xs, vs, k=3) if len(xs) >= 4 else (lambda z: np.interp(z, xs, vs))


def _derivatives(u, dx):
    """u_x and u_xx: fourth-order central differences inside, second order next to
    the edges (one-sided at the edges themselves, where the equation is not used)."""
    u_x = np.gradient(u, dx, edge_order=2)
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    if len(u) >= 5:
        u_x[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dx)
        u_xx[2:-2] = (-u[:-4] + 16 * u[1:-3] - 30 * u[2:-2] + 16 * u[3:-1] - u[4:]) / (12 * dx ** 2)
    return u_x, u_xx


def simulate(theta, E, X):
    """Solve the equation (method of lines) and return u at the points X.
    After the last fitted time the edge values are held at their last value."""
    P, F, dic, dbl, dbr = _unpack(theta, E)
    nx = E["nx"]
    xf = np.linspace(E["xlo"], E["xhi"], nx)
    dx = xf[1] - xf[0]
    left = lambda tt: np.interp(tt, *E["bl"]) + np.interp(tt, E["tn"], dbl)
    right = lambda tt: np.interp(tt, *E["br"]) + np.interp(tt, E["tn"], dbr)
    # the start profile from the data, run out to the edge values at the start time
    # where the earliest data stop short of an edge (e.g. a held-out corner point):
    # held flat instead, it put a step next to the edge the corrections could not remove
    gx, gv = E["ic"]
    if gx[0] > E["xlo"]:
        gx, gv = np.concatenate([[E["xlo"]], gx]), np.concatenate([[left(E["t0"])], gv])
    if gx[-1] < E["xhi"]:
        gx, gv = np.concatenate([gx, [E["xhi"]]]), np.concatenate([gv, [right(E["t0"])]])
    # the correction ends at the edge values: a smoothed profile that misses an edge
    # value (a narrow pulse flattened by the smoothing) no longer leaves a ramp there
    base = np.interp(xf, gx, gv)
    lo, hi = left(E["t0"]) - base[0], right(E["t0"]) - base[-1]
    u0 = base + _correction(E, dic, lo, hi)(xf)
    sparsity = diags([1] * 5, [-2, -1, 0, 1, 2], shape=(nx - 2, nx - 2), dtype=float)

    def f(tt, ui):
        u = np.concatenate([[left(tt)], ui, [right(tt)]])
        u_x, u_xx = _derivatives(u, dx)
        rate = rhs(xf, tt, u, u_x, u_xx, P, F)
        return np.broadcast_to(rate, u.shape)[1:-1]

    tq = X[:, TI]
    ts = np.unique(np.concatenate([[E["t0"]], tq[tq >= E["t0"]]]))
    if len(ts) == 1:
        ts = np.array([E["t0"], E["t0"] + 1e-12])
        U = np.column_stack([u0, u0])
    else:
        try:
            with np.errstate(all="ignore"):
                sol = solve_ivp(f, (ts[0], ts[-1]), u0[1:-1], t_eval=ts, method="BDF", rtol=RTOL,
                                atol=1e-9 * (float(np.max(np.abs(u0))) or 1.0),
                                jac_sparsity=sparsity)  # each point sees two neighbours a side
        except Exception:      # e.g. a singular matrix for a wild guess: a failed solve
            return None
        if not sol.success or sol.y.shape[1] != len(ts):
            return None
        U = np.vstack([left(ts), sol.y, right(ts)])
    interp = RegularGridInterpolator((xf, ts), U, bounds_error=False, fill_value=None)
    return interp(np.column_stack([X[:, XI], np.maximum(tq, E["t0"])]))


def _regression_start(X, y):
    """Starting constants from the data's own derivatives, without solving: a smooth
    Chebyshev surface through the points (holes are fine), then the equation fitted to
    its u, u_x, u_xx and u_t. Constants can be any size and sign: screen 24 decades,
    zoom twice, refine. Returns None when it cannot be built."""
    n = len(PARAMS)
    if n == 0 or UFUNCS:
        return None
    x, t = X[:, XI], X[:, TI]
    nxu, ntu = len(np.unique(x)), len(np.unique(t))
    dx, dt = int(min(8, nxu - 2)), int(min(6, ntu - 2))
    if dx < 2 or dt < 1:
        return None
    C = np.polynomial.chebyshev
    sx, st = 2 / (x.max() - x.min()), 2 / (t.max() - t.min())
    zx, zt = (x - x.min()) * sx - 1, (t - t.min()) * st - 1
    V = C.chebvander2d(zx, zt, [dx, dt])
    coef = np.linalg.solve(V.T @ V + 1e-8 * len(y) * np.eye(V.shape[1]), V.T @ y).reshape(dx + 1, dt + 1)
    u = C.chebval2d(zx, zt, coef)
    ux = C.chebval2d(zx, zt, C.chebder(coef, 1, axis=0)) * sx
    uxx = C.chebval2d(zx, zt, C.chebder(coef, 2, axis=0)) * sx ** 2
    ut = C.chebval2d(zx, zt, C.chebder(coef, 1, axis=1)) * st

    def loss(vals):
        P = {q: vals[:, i:i + 1] for i, q in enumerate(PARAMS)}
        with np.errstate(all="ignore"):
            r = np.broadcast_to(rhs(x, t, u, ux, uxx, P, {}), (len(vals), len(x))) - ut
            v = np.mean(r ** 2, axis=1)
        return np.where(np.isfinite(v), v, np.inf)

    rng = np.random.default_rng(0)

    def draw(center, half, m):
        return (np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)
                * 10 ** (center + half * (2 * rng.random((m, n)) - 1)))

    cand = np.vstack([np.ones((1, n)), draw(np.zeros(n), 12.0, 16384)])
    best = cand[np.argmin(loss(cand))]
    for half in (1.5, 0.3):
        cand = np.vstack([best, np.sign(best) * np.abs(draw(np.log10(np.abs(best)), half, 4096))])
        best = cand[np.argmin(loss(cand))]
    try:
        r = least_squares(lambda v: np.where(np.isfinite(d := np.broadcast_to(
            rhs(x, t, u, ux, uxx, dict(zip(PARAMS, v)), {}), x.shape) - ut), d, 1e6), best, x_scale="jac")
        return r.x if np.all(np.isfinite(r.x)) else best
    except Exception:
        return best


def train(X, y, seed=0, domain=None):
    E = edges(X, y)
    edge0 = np.zeros(len(E["xn"]) + 2 * len(E["tn"]))     # no correction to start with
    scale = float(np.std(y)) or 1.0
    rng = np.random.default_rng(seed)
    n = len(PARAMS)
    if n == 0:
        guesses = [np.ones(0)]
    elif n == 1:
        guesses = [np.array([v]) for v in 10 ** np.linspace(-2, 1, 7)]
    else:
        guesses = [np.ones(n), 0.1 * np.ones(n), 10 * np.ones(n)]
        while len(guesses) < SCREEN:
            guesses.append(10 ** rng.uniform(-2, 1, n))
    reg = _regression_start(X, y)
    if reg is not None:
        guesses.insert(0, reg)

    def resid(theta):
        sim = simulate(theta, E, X)
        if sim is None or not np.all(np.isfinite(sim)):
            return np.full(len(y), 1e3)
        return (sim - y) / scale

    # 1. constants only, with the start profile and edges as the data has them: score
    #    every guess with one solve, then fit from the best
    def cost(g):
        theta0 = np.concatenate([g, np.zeros(len(UFUNCS) * (UDEG + 1))])
        return float(np.sum(resid(np.concatenate([theta0, edge0])) ** 2))

    best = None
    for g in sorted(guesses, key=cost)[:1]:
        theta0 = np.concatenate([g, np.zeros(len(UFUNCS) * (UDEG + 1))])
        if theta0.size == 0:          # every constant known: only the corrections are fitted
            break
        try:
            r = least_squares(lambda th: resid(np.concatenate([th, edge0])), theta0,
                              x_scale="jac", diff_step=DIFF_STEP, max_nfev=200)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None and len(PARAMS) + len(UFUNCS) == 0:
        theta = edge0.copy()
    elif best is None:
        raise RuntimeError("no starting guess converged")
    else:
        theta = np.concatenate([best.x, edge0])
    # 2. alternate: the corrections by a direct linear solve (exact when the equation is
    #    linear in u, one Gauss-Newton step otherwise), then the constants with them held
    k, h = len(theta) - len(edge0), 0.01 * scale
    current = float(np.sum(resid(theta) ** 2))
    for _ in range(EDGE_ROUNDS):
        r0 = resid(theta)
        J = np.column_stack([(resid(theta + h * np.eye(len(theta))[i]) - r0) / h
                             for i in range(k, len(theta))])
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        for frac in (1.0, 0.5, 0.25):
            trial = theta.copy()
            trial[k:] += frac * step
            c = float(np.sum(resid(trial) ** 2))
            if c < current:
                theta, current = trial, c
                break
        else:
            break
        if k:
            try:
                r = least_squares(lambda th: resid(np.concatenate([th, theta[k:]])), theta[:k],
                                  x_scale="jac", diff_step=DIFF_STEP, max_nfev=50)
                if 2 * r.cost < current:
                    theta[:k], current = r.x, 2 * r.cost
            except Exception:
                pass
    def jacobian(theta, r0=None):
        r0 = resid(theta) if r0 is None else r0
        steps = [DIFF_STEP * max(abs(v), 1e-8) if i < k else h for i, v in enumerate(theta)]
        return np.column_stack([(resid(theta + s * np.eye(len(theta))[i]) - r0) / s
                                for i, s in enumerate(steps)])

    # 3. everything together: the constants and the start profile are strongly coupled
    #    (a narrow pulse that moves), so alternating alone can stall far from the best fit
    if k and len(theta) > k and EDGE_ROUNDS and JOINT_STEPS:
        try:
            r = least_squares(resid, theta, jac=lambda th: jacobian(th), x_scale="jac",
                              max_nfev=JOINT_STEPS)
            if 2 * r.cost < current:
                theta, current = r.x, 2 * r.cost
        except Exception:
            pass

    # standard errors of the constants, with the start and edge corrections as nuisance
    # parameters (their uncertainty widens the constants' errors)
    def errors(theta):
        r0 = resid(theta)
        return standard_errors(jacobian(theta, r0), r0)[:len(PARAMS)]

    se = errors(theta)
    # grid check: refit the constants on a finer grid, and keep refining while they move
    # by more than GRID_TOL standard errors. The finest fit is kept; its last move is the
    # solver's own error, and is added to the standard errors.
    solver, refined = np.zeros(len(PARAMS)), False
    while len(PARAMS) and 2 * E["nx"] - 1 <= NX_MAX:
        coarse = theta[:len(PARAMS)].copy()
        E = dict(E, nx=2 * E["nx"] - 1)              # resid() now solves on the finer grid
        try:
            r = least_squares(lambda th: resid(np.concatenate([th, theta[k:]])), theta[:k],
                              x_scale="jac", diff_step=DIFF_STEP, max_nfev=30)
        except Exception:
            break
        theta[:k], refined = r.x, True
        solver = np.abs(theta[:len(PARAMS)] - coarse)
        if np.all(solver <= GRID_TOL * se):
            break
    if refined:
        se = errors(theta)
    se = np.sqrt(np.asarray(se, float) ** 2 + solver ** 2)
    state = {"theta": theta, "edges": E, "se": dict(zip(PARAMS, map(float, se))),
             "solver_error": dict(zip(PARAMS, map(float, solver)))}
    note = _outside_note(X, domain)
    if note:
        state["note"] = note
    return state


def _outside_note(X, domain):
    """Where predictions are wanted outside the trained times, say what the fit does
    there: the edge values are known only from the data, so after the last training
    time they are held at their last fitted values, and before the first the start
    profile is returned. Neither is the equation's prediction."""
    if domain is None:
        return None
    t = X[:, TI]
    span = float(t.max() - t.min()) or 1.0
    lo, hi = float(domain[0][TI]), float(domain[1][TI])
    parts = []
    if hi > t.max() + 1e-9 * span:
        parts.append(f"after t = {t.max():.4g}, the last time it trained on, the edge values "
                     "are not known, so they are held at their last fitted values; predictions "
                     "there are only as good as that guess")
    if lo < t.min() - 1e-9 * span:
        parts.append(f"before t = {t.min():.4g}, the first time it trained on, it returns the "
                     "start profile")
    return ("Outside the trained times: " + "; ".join(parts) + ".") if parts else None


def predict(state, X):
    sim = simulate(state["theta"], state["edges"], X)
    return np.full(len(X), np.nan) if sim is None else sim


def fitted_params(state):
    return {p: float(v) for p, v in zip(PARAMS, state["theta"][:len(PARAMS)])}


def uncertainty(state):
    """Standard error of each fitted constant (inf = not determined by the data)."""
    return dict(state.get("se", {}))


def learned_function(state, name, z):
    return _split(state["theta"])[1][name](np.asarray(z, float))
'''

# --------------------------------------------------------------------------
# Physics-informed: a smooth model trained on data + the equation's residual.
# --------------------------------------------------------------------------

BODY_MLP = '''
class Body(torch.nn.Module):
    """Fully connected network: inputs scaled to [-1, 1], 3 hidden layers of 32 tanh units."""

    def __init__(self, d, width=32, depth=3):
        super().__init__()
        layers, n = [], d
        for _ in range(depth):
            layers += [torch.nn.Linear(n, width), torch.nn.Tanh()]
            n = width
        layers.append(torch.nn.Linear(n, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)
'''

BODY_CHEB = '''
DEG = <<DEG>>  # polynomial degree in each variable


class Body(torch.nn.Module):
    """Chebyshev polynomial of degree DEG in each variable (all products across variables)."""

    def __init__(self, d):
        super().__init__()
        grids = [torch.arange(DEG + 1)] * d
        idx = torch.cartesian_prod(*grids) if d > 1 else grids[0][:, None]
        self.register_buffer("idx", idx)
        self.coef = torch.nn.Parameter(torch.zeros(len(idx)))

    def features(self, z):
        T = [torch.ones_like(z), z]
        for _ in range(2, DEG + 1):
            T.append(2 * z * T[-1] - T[-2])
        T = torch.stack(T[:DEG + 1], -1)  # (N, d, DEG+1)
        out = torch.ones(z.shape[0], len(self.idx), dtype=z.dtype)
        for j in range(z.shape[1]):
            out = out * T[:, j, self.idx[:, j]]
        return out

    def forward(self, z):
        return self.features(z) @ self.coef

    def matrix(self, z, orders):
        """Fixed matrix M with (d/dz)^orders of the polynomial at z = M @ coef, for orders
        given per variable. Uses T_n^(k) = 2z T_(n-1)^(k) + 2k T_(n-1)^(k-1) - T_(n-2)^(k)."""
        out = torch.ones(z.shape[0], len(self.idx), dtype=z.dtype)
        for j, k in enumerate(orders):
            x = z[:, j]
            T = [torch.ones_like(x), x]
            for _ in range(2, DEG + 1):
                T.append(2 * x * T[-1] - T[-2])
            T = T[:DEG + 1]
            for order in range(1, k + 1):
                D = [torch.zeros_like(x), torch.ones_like(x) if order == 1 else torch.zeros_like(x)]
                for n in range(2, DEG + 1):
                    D.append(2 * x * D[n - 1] + 2 * order * T[n - 1] - D[n - 2])
                T = D[:DEG + 1]
            out = out * torch.stack(T, -1)[:, self.idx[:, j]]
        return out

    def fit_data(self, z, target, ridge=1e-6):
        """Closed-form smoothing least squares start (before the equation is switched on).
        High degrees are penalised harder, so the start follows the trend, not the noise."""
        A = self.features(z)
        total_degree = self.idx.sum(1).to(A.dtype)
        n = A.shape[0]
        lhs = A.T @ A / n + torch.diag(ridge * (1.0 + total_degree) ** 4)
        with torch.no_grad():
            self.coef.copy_(torch.linalg.solve(lhs, A.T @ target / n))
'''

PHYSICS = '''
import torch

DERIVS = <<DERIVS>>  # derivatives the equation needs, by variable
N_COLLOC = <<NCOL>>  # points where the equation is enforced
STEPS_DATA = 1000    # warm-up: fit the data alone
STEPS = <<STEPS>>    # Adam steps on data + equation
LBFGS_ITERS = <<LBFGS>>
WEIGHT = 1.0         # weight of the equation term (both terms are normalised)
SCREEN = 4096        # starting values tried for the constants (on the warm-up curve)
DOMAIN = None        # where the equation is enforced: [[lows], [highs]] per variable.
                     # None = the range of the data. Widen it to predict beyond the data.

<<RESIDUAL_FN>>

<<BODY>>

class UFunc(torch.nn.Module):
    """Unknown function of one argument: small network, 2 x 16 tanh."""

    def __init__(self, center, spread):
        super().__init__()
        self.center, self.spread, self.out = center, spread, 1.0
        self.net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(),
                                       torch.nn.Linear(16, 16), torch.nn.Tanh(),
                                       torch.nn.Linear(16, 1))

    def forward(self, a):
        return self.out * self.net(((a - self.center) / self.spread)[..., None]).squeeze(-1)


class Model(torch.nn.Module):
    def __init__(self, lo, hi, ym, ys):
        super().__init__()
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.ym, self.ys = ym, ys
        self.body = Body(len(INDEP))
        self.P = torch.nn.ParameterDict({p: torch.nn.Parameter(torch.tensor(1.0)) for p in PARAMS})
        self.F = torch.nn.ModuleDict({f: UFunc(ym, ys) for f in UFUNCS})

    def scaled(self, X):
        return 2 * (X - self.lo) / (self.hi - self.lo) - 1

    def forward(self, X):
        return self.ym + self.ys * self.body(self.scaled(X))


def values(model, X):
    """Target and its derivatives at X, by automatic differentiation."""
    X = X.detach().clone().requires_grad_(True)
    u = model(X)
    V = {v: X[:, i] for i, v in enumerate(INDEP)}
    V[TARGET] = u
    cache = {(): u}
    for name, seq in DERIVS.items():
        for k in range(1, len(seq) + 1):
            key = seq[:k]
            if key not in cache:
                g = torch.autograd.grad(cache[key[:-1]].sum(), X, create_graph=True)[0]
                cache[key] = g[:, INDEP.index(key[-1])]
        V[name] = cache[seq]
    return V


def _lbfgs(params, closure, iters):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def step():
        opt.zero_grad()
        loss = closure()
        loss.backward()
        return loss

    opt.step(step)


def _train(X, y, seed=0, domain=None):
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    domain = domain if domain is not None else DOMAIN
    if domain is None:
        lo, hi = Xt.min(0).values, Xt.max(0).values
    else:
        lo, hi = (torch.tensor(np.asarray(v, float), dtype=torch.float64) for v in domain)
        lo, hi = torch.minimum(lo, Xt.min(0).values), torch.maximum(hi, Xt.max(0).values)
    hi = torch.where(hi > lo, hi, lo + 1.0)
    ym, ys = float(y.mean()), float(y.std()) or 1.0
    model = Model(lo, hi, ym, ys)
    C = lo + (hi - lo) * torch.rand(N_COLLOC, len(INDEP))

    # A body that is linear in its coefficients (the polynomial): its values and
    # derivatives at fixed points are fixed matrices times the coefficients. Build them
    # once instead of differentiating through the body at every step.
    linear = hasattr(model.body, "matrix")
    if linear:
        def mats(Xp):
            z = model.scaled(Xp)
            out = {(): model.body.matrix(z, [0] * len(INDEP))}
            for seq in DERIVS.values():
                orders = [seq.count(v) for v in INDEP]
                chain = float(torch.prod((2 / (hi - lo)) ** torch.tensor(orders, dtype=lo.dtype)))
                out[seq] = chain * model.body.matrix(z, orders)
            return out
        Md, Mc = mats(Xt), mats(C)

    def data_loss():
        pred = model.ym + model.ys * (Md[()] @ model.body.coef) if linear else model(Xt)
        return torch.mean((pred - yt) ** 2) / ys ** 2

    def colloc_values():
        if not linear:
            return values(model, C)
        c = model.body.coef
        V = {v: C[:, i] for i, v in enumerate(INDEP)}
        V[TARGET] = model.ym + model.ys * (Mc[()] @ c)
        for name, seq in DERIVS.items():
            V[name] = model.ys * (Mc[seq] @ c)
        return V

    # 1. warm-up on the data alone, so the derivatives have a sensible size
    if hasattr(model.body, "fit_data"):
        model.body.fit_data(model.scaled(Xt), (yt - ym) / ys)
    else:
        opt = torch.optim.Adam(model.body.parameters(), lr=3e-3)
        for _ in range(STEPS_DATA):
            opt.zero_grad()
            loss = data_loss()
            loss.backward()
            opt.step()

    # 2. scale for the equation term: typical size of the derivatives it contains
    V = {k: v.detach() for k, v in values(model, C).items()}
    res_scale = max(float(torch.sqrt(torch.mean(V[d] ** 2))) for d in DERIVS) or 1.0
    for f in model.F.values():
        f.out = res_scale

    # 3. starting constants: best fit of the equation to the warm-up curve. Constants can
    #    be any size and either sign (a carrying capacity of 665, a circuit's 1/(LC) of
    #    5e5), so first screen many values on the fixed warm-up curve (cheap: no training),
    #    over 24 decades (1e-12 to 1e12), then zoom twice around the best, and start from
    #    the best.
    if PARAMS:
        gen = torch.Generator().manual_seed(seed)
        k = len(PARAMS)

        def screen_loss(vals):
            P = {p: vals[:, i:i + 1] for i, p in enumerate(PARAMS)}
            with torch.no_grad():
                r = residual(V, P, model.F) / res_scale
                loss = torch.mean(torch.broadcast_to(r, (len(vals), N_COLLOC)) ** 2, dim=1)
            return torch.where(torch.isfinite(loss), loss, torch.tensor(float("inf")))

        def draw(log_center, half_width, n):
            mag = log_center + half_width * (2 * torch.rand(n, k, generator=gen) - 1)
            sign = torch.where(torch.rand(n, k, generator=gen) < 0.5, -1.0, 1.0)
            return sign * 10 ** mag

        cand = torch.cat([torch.ones(1, k), draw(torch.zeros(k), 12.0, SCREEN)])
        best = cand[torch.argmin(screen_loss(cand))]
        for half in (1.5, 0.3):
            zoom = torch.sign(best) * draw(torch.log10(best.abs()), half, SCREEN).abs()
            cand = torch.cat([best[None], zoom])
            best = cand[torch.argmin(screen_loss(cand))]
        with torch.no_grad():
            for i, p in enumerate(PARAMS):
                model.P[p].copy_(best[i])
    if PARAMS or UFUNCS:
        _lbfgs(list(model.P.parameters()) + list(model.F.parameters()),
               lambda: torch.mean((residual(V, model.P, model.F) / res_scale) ** 2), 200)

    # 4. train everything on data + equation
    def total():
        r = residual(colloc_values(), model.P, model.F) / res_scale
        return data_loss() + WEIGHT * torch.mean(r ** 2)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = total()
        loss.backward()
        opt.step()
        sched.step()
    if LBFGS_ITERS:
        _lbfgs(model.parameters(), total, LBFGS_ITERS)
    return {"model": model}

def train(X, y, seed=0, domain=None):
    """Train on one CPU thread: with several, sums are added in an order that depends on
    the machine's core count and training amplifies the rounding, so the same data and
    seed could give different answers on different machines. One thread is also faster
    at these sizes. The caller's thread and dtype settings are restored afterwards."""
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)   # only while training: never changes the caller's torch
    try:
        return _train(X, y, seed, domain)
    finally:
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)


def predict(state, X):
    with torch.no_grad():
        return state["model"](torch.tensor(X, dtype=torch.float64)).numpy()


def fitted_params(state):
    return {p: float(v.detach()) for p, v in state["model"].P.items()}


def learned_function(state, name, z):
    with torch.no_grad():
        return state["model"].F[name](torch.tensor(np.asarray(z, float))).numpy()
'''

# --------------------------------------------------------------------------
# Data only (no equation): the baselines that show whether the equation helped.
# --------------------------------------------------------------------------

NN_DATA = '''
import torch

STEPS = <<STEPS>>
LBFGS_ITERS = <<LBFGS>>
''' + BODY_MLP + '''

def _train(X, y, seed=0):
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    lo, hi = Xt.min(0).values, Xt.max(0).values
    hi = torch.where(hi > lo, hi, lo + 1.0)
    ym, ys = float(y.mean()), float(y.std()) or 1.0
    body = Body(len(INDEP))

    def loss_fn():
        z = 2 * (Xt - lo) / (hi - lo) - 1
        return torch.mean((ym + ys * body(z) - yt) ** 2) / ys ** 2

    opt = torch.optim.Adam(body.parameters(), lr=3e-3)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    if LBFGS_ITERS:
        lb = torch.optim.LBFGS(body.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        lb.step(closure)
    return {"body": body, "lo": lo, "hi": hi, "ym": ym, "ys": ys}

def train(X, y, seed=0):
    """Train on one CPU thread: with several, sums are added in an order that depends on
    the machine's core count and training amplifies the rounding, so the same data and
    seed could give different answers on different machines. One thread is also faster
    at these sizes. The caller's thread and dtype settings are restored afterwards."""
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)   # only while training: never changes the caller's torch
    try:
        return _train(X, y, seed)
    finally:
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)


def predict(state, X):
    with torch.no_grad():
        z = 2 * (torch.tensor(X, dtype=torch.float64) - state["lo"]) / (state["hi"] - state["lo"]) - 1
        return (state["ym"] + state["ys"] * state["body"](z)).numpy()


def fitted_params(state):
    return {}


def learned_function(state, name, z):
    raise ValueError("this model does not use the equation")
'''

SPLINE_DATA = '''
from scipy.interpolate import RBFInterpolator, make_smoothing_spline


def train(X, y, seed=0, folds=5):
    if X.shape[1] == 1:
        # cubic smoothing spline, smoothness chosen by generalised cross-validation
        t, inv = np.unique(X[:, 0], return_inverse=True)
        if len(t) < 5:
            raise ValueError(f"the smoothing spline needs at least 5 distinct values of the "
                             f"variable to train on; got {len(t)}")
        ym = np.bincount(inv, y) / np.bincount(inv)
        # on a 0-1 scale: the smoothness choice must not depend on the time unit
        lo, span = float(t[0]), float(t[-1] - t[0])
        return {"kind": "1d", "spline": make_smoothing_spline((t - lo) / span, ym), "lo": lo, "span": span}
    # thin-plate spline, smoothing chosen by cross-validation
    lo, hi = X.min(0), X.max(0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    z = (X - lo) / (hi - lo)
    fold = np.random.default_rng(seed).permutation(len(y)) % folds
    best = (np.inf, 0.0)
    for s in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        err = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            f = RBFInterpolator(z[tr], y[tr], kernel="thin_plate_spline", smoothing=s)
            err += np.sum((f(z[te]) - y[te]) ** 2)
        best = min(best, (err, s))
    f = RBFInterpolator(z, y, kernel="thin_plate_spline", smoothing=best[1])
    return {"kind": "nd", "f": f, "lo": lo, "hi": hi}


def predict(state, X):
    if state["kind"] == "1d":
        return state["spline"]((X[:, 0] - state["lo"]) / state["span"])
    return state["f"]((X - state["lo"]) / (state["hi"] - state["lo"]))


def fitted_params(state):
    return {}


def learned_function(state, name, z):
    raise ValueError("this model does not use the equation")
'''

POLY_DATA = '''
MAX_DEG = <<MAXDEG>>  # highest degree tried in each variable


def _features(z, deg):
    T = np.polynomial.chebyshev.chebvander(z[:, 0], deg)
    for j in range(1, z.shape[1]):
        Tj = np.polynomial.chebyshev.chebvander(z[:, j], deg)
        T = (T[:, :, None] * Tj[:, None, :]).reshape(len(z), -1)
    return T


def _solve(A, b, ridge=1e-8):
    return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)


def train(X, y, seed=0, folds=10):
    lo, hi = X.min(0), X.max(0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    z = 2 * (X - lo) / (hi - lo) - 1
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    # points on the edge of the data stay in every training fold: otherwise the
    # check would score extrapolation, where high degrees always look bad
    edge = np.any((X == X.min(0)) | (X == X.max(0)), axis=1)
    if (~edge).sum() >= folds:     # enough inner points to check on; else check on all points
        fold[edge] = -1
    else:
        edge[:] = False
    best = (np.inf, 0)
    for deg in range(0, MAX_DEG + 1):
        if (deg + 1) ** X.shape[1] >= 0.8 * (~edge).sum() * (folds - 1) / folds + edge.sum():
            break
        err = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            if not te.any():
                continue
            c = _solve(_features(z[tr], deg), y[tr])
            err += np.sum((_features(z[te], deg) @ c - y[te]) ** 2)
        best = min(best, (err, deg))
    deg = best[1]
    return {"deg": deg, "lo": lo, "hi": hi, "coef": _solve(_features(z, deg), y)}


def predict(state, X):
    z = 2 * (X - state["lo"]) / (state["hi"] - state["lo"]) - 1
    return _features(z, state["deg"]) @ state["coef"]


def fitted_params(state):
    return {}


def learned_function(state, name, z):
    raise ValueError("this model does not use the equation")
'''
