"""Stage A of TESTPLAN.md: the fresh benchmark problems.

Written and committed BEFORE the first stage A run (2026-11-16). None of these
problems, constants or settings were used while developing equationnet, and none
may be tuned on: if a fix follows from stage A, the rerun uses new seeds.

Each problem gives: the equation as a user would type it; a data generator
(seed -> columns, with noise); the exact solution `truth(X)` for scoring on a dense
grid; the true constants; whether it is built to be identifiable; and the expected
outcome, written down here before running (reported under A7, no bar).
"""

import numpy as np
from scipy.integrate import solve_ivp


def _ode(rhs, y0, T, rtol=1e-11):
    return solve_ivp(rhs, (0, T), y0, dense_output=True, rtol=rtol, atol=1e-13, method="DOP853")


def _noisy(clean, level, seed, scale=None):
    s = np.std(clean) if scale is None else scale
    return clean + level * s * np.random.default_rng(seed).standard_normal(clean.shape)


def _grid(x, t):
    XX, TT = np.meshgrid(x, t)
    return XX.ravel(), TT.ravel()


P = {}

# 1. Gompertz growth, carrying capacity far from 1 (2500), 25 points, 2% noise.
#    Expect: classical best; K and r within 3 SE; equation models forecast well.
_g = _ode(lambda t, y: [0.35 * y[0] * np.log(2500 / y[0])], [40.0], 20)
P["gompertz"] = dict(
    equation="u_t = r*u*log(K/u)", true={"r": 0.35, "K": 2500.0}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_g.sol(t)[0], 0.02, s)})(np.linspace(0, 20, 25)),
    truth=lambda X: _g.sol(X[:, 0])[0])

# 2. Second-order decay u_t = -k*u^2, k = 4e-4 (small constant), 20 points, 1% noise.
_d2 = _ode(lambda t, y: [-4e-4 * y[0] ** 2], [1500.0], 10)
P["quadratic_decay"] = dict(
    equation="u_t = -k*u^2", true={"k": 4e-4}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_d2.sol(t)[0], 0.01, s)})(np.linspace(0, 10, 20)),
    truth=lambda X: _d2.sol(X[:, 0])[0])

# 3. Nonlinear pendulum, amplitude 2.5 rad, w = g/L = 19.62, 50 points over 6 s.
#    Expect: classical best; a linear small-angle model would be wrong here.
_p = _ode(lambda t, y: [y[1], -19.62 * np.sin(y[0])], [2.5, 0.0], 6)
P["pendulum"] = dict(
    equation="u_tt = -w*sin(u)", true={"w": 19.62}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_p.sol(t)[0], 0.02, s)})(np.linspace(0, 6, 50)),
    truth=lambda X: _p.sol(X[:, 0])[0])

# 4. Damped Duffing oscillator, UNEVENLY spaced times (60 random times in 15 s), 3% noise.
_du = _ode(lambda t, y: [y[1], -0.15 * y[1] - 2.0 * y[0] - 8.0 * y[0] ** 3], [1.2, 0.0], 15)
P["duffing_uneven"] = dict(
    equation="u_tt + c*u_t + a*u + b*u^3 = 0", true={"c": 0.15, "a": 2.0, "b": 8.0}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_du.sol(t)[0], 0.03, s)})(
        np.sort(np.random.default_rng(100 + s).uniform(0, 15, 60))),
    truth=lambda X: _du.sol(X[:, 0])[0])

# 5. Van der Pol, mu = 1.5, 80 points over 20 s, 5% noise.
_v = _ode(lambda t, y: [y[1], 1.5 * (1 - y[0] ** 2) * y[1] - y[0]], [0.5, 0.0], 20)
P["van_der_pol"] = dict(
    equation="u_tt - mu*(1 - u^2)*u_t + u = 0", true={"mu": 1.5}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_v.sol(t)[0], 0.05, s)})(np.linspace(0, 20, 80)),
    truth=lambda X: _v.sol(X[:, 0])[0])

# 6. Stiff forced relaxation u_t = -k*(u - cos(t)), k = 50, 40 points, 1% noise.
_st = solve_ivp(lambda t, y: [-50 * (y[0] - np.cos(t))], (0, 6), [2.0], dense_output=True,
                rtol=1e-11, atol=1e-13, method="Radau")
P["stiff_forced"] = dict(
    equation="u_t = -k*(u - cos(t))", true={"k": 50.0}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_st.sol(t)[0], 0.01, s)})(np.linspace(0, 6, 40)),
    truth=lambda X: _st.sol(X[:, 0])[0])

# 7. Very sparse (10 points), time in seconds over 40 microseconds, values in the
#    millions, k = 1.2e5 /s, 0.5% noise.
P["microseconds_sparse"] = dict(
    equation="u_t = -k*u", true={"k": 1.2e5}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(4.8e6 * np.exp(-1.2e5 * t), 0.005, s)})(
        np.linspace(0, 4e-5, 10)),
    truth=lambda X: 4.8e6 * np.exp(-1.2e5 * X[:, 0]))

# 8. Advection-diffusion of a Gaussian pulse, v = 0.8, D = 0.02, 21 x 11 grid, 1% noise.
def _advdiff(X):
    x, t = X[:, 0], X[:, 1]
    w = 0.01 + 4 * 0.02 * t                       # 4*D*t + initial width^2 (sigma0^2 = 0.005)
    return np.sqrt(0.01 / w) * np.exp(-(x - 0.5 - 0.8 * t) ** 2 / w)


P["advection_diffusion"] = dict(
    equation="u_t + v*u_x = D*u_xx", true={"v": 0.8, "D": 0.02}, identifiable=True,
    data=lambda s: (lambda x, t: {"x": x, "t": t, "u": _noisy(_advdiff(np.column_stack([x, t])), 0.01, s)})(
        *_grid(np.linspace(0, 2, 21), np.linspace(0, 1, 11))),
    truth=_advdiff)

# 9. Fisher-KPP travelling front (exact solution), D = 0.01, r = 2, 21 x 11 grid, 1% noise.
def _fisher(X):
    x, t = X[:, 0], X[:, 1]
    D, r = 0.01, 2.0
    return 1.0 / (1 + np.exp(np.sqrt(r / (6 * D)) * (x - 1.0) - 5 * r / 6 * t)) ** 2


P["fisher_kpp"] = dict(
    equation="u_t = D*u_xx + r*u*(1 - u)", true={"D": 0.01, "r": 2.0}, identifiable=True,
    data=lambda s: (lambda x, t: {"x": x, "t": t, "u": _noisy(_fisher(np.column_stack([x, t])), 0.01, s)})(
        *_grid(np.linspace(0, 2, 21), np.linspace(0, 1, 11))),
    truth=_fisher)

# 10. Viscous Burgers (exact Cole-Hopf solution), nu = 0.05, 21 x 11 grid, 1% noise.
def _burgers(X):
    x, t = X[:, 0], X[:, 1]
    nu, a = 0.05, 2.0
    e = np.exp(-np.pi ** 2 * nu * t)
    return 2 * nu * np.pi * e * np.sin(np.pi * x) / (a + e * np.cos(np.pi * x))


P["burgers"] = dict(
    equation="u_t + u*u_x = nu*u_xx", true={"nu": 0.05}, identifiable=True,
    data=lambda s: (lambda x, t: {"x": x, "t": t, "u": _noisy(_burgers(np.column_stack([x, t])), 0.01, s)})(
        *_grid(np.linspace(0, 1, 21), np.linspace(0, 1, 11))),
    truth=_burgers)

# 11. Heat equation with non-zero, different edge values, D = 0.05, 0.1% noise.
def _heat_edges(X):
    x, t = X[:, 0], X[:, 1]
    return 1 + 0.5 * x + np.exp(-0.05 * np.pi ** 2 * t) * np.sin(np.pi * x)


P["heat_edges"] = dict(
    equation="u_t = D*u_xx", true={"D": 0.05}, identifiable=True,
    data=lambda s: (lambda x, t: {"x": x, "t": t, "u": _noisy(_heat_edges(np.column_stack([x, t])), 0.001, s)})(
        *_grid(np.linspace(0, 1, 11), np.linspace(0, 2, 11))),
    truth=_heat_edges)

# 12. Dense and very noisy: logistic, 500 points, 20% noise, r = 0.9, K = 40.
def _logistic(t):
    return 40 / (1 + (40 / 1.5 - 1) * np.exp(-0.9 * t))


P["logistic_dense_noisy"] = dict(
    equation="u_t = r*u*(1 - u/K)", true={"r": 0.9, "K": 40.0}, identifiable=True,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(_logistic(t), 0.20, s)})(np.linspace(0, 15, 500)),
    truth=lambda X: _logistic(X[:, 0]))

# 13. NOT identifiable by construction: only a + b is determined.
#     Expect: "not pinned down" for a and b in every run (bar A6).
P["sum_of_rates"] = dict(
    equation="u_t = -(a + b)*u", true={"a": None, "b": None}, identifiable=False,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(3 * np.exp(-0.6 * t), 0.01, s)})(np.linspace(0, 5, 30)),
    truth=lambda X: 3 * np.exp(-0.6 * X[:, 0]))

# 14. NOT identifiable by construction: a single exponential fitted with a second-order
#     equation; only the combination with root -0.5 (k = 0.5*c - 0.25) is determined.
#     Expect: "not pinned down" for c and k in every run (bar A6).
P["one_exponential_second_order"] = dict(
    equation="u_tt + c*u_t + k*u = 0", true={"c": None, "k": None}, identifiable=False,
    data=lambda s: (lambda t: {"t": t, "u": _noisy(2 * np.exp(-0.5 * t), 0.01, s)})(np.linspace(0, 8, 30)),
    truth=lambda X: 2 * np.exp(-0.5 * X[:, 0]))

SEEDS = [11, 12, 13]                          # new seeds, not used anywhere before
COVERAGE_FAMILIES = ["gompertz", "duffing_uneven", "advection_diffusion", "burgers"]
COVERAGE_SEEDS = list(range(500, 520))        # 20 seeds for bar A4

# Run 2 (after the run-1 fixes, 2026-11-17): fresh seeds, fixed before running. Run 1's
# seeds (11-13, 500-519) were used to diagnose and fix, so they no longer count.
SEEDS_RUN2 = [21, 22, 23]
COVERAGE_SEEDS_RUN2 = list(range(600, 620))

# Run 3 (after the run-2 fixes, 2026-11-17): fresh seeds, fixed before running. Run 2's
# seeds (21-23, 600-619) and the diagnosis seeds 900-939 were used to fix, so they no
# longer count.
SEEDS_RUN3 = [31, 32, 33]
COVERAGE_SEEDS_RUN3 = list(range(700, 720))

# Run 4 (after the stiff start-value fix found by the second preflight recheck,
# 2026-11-17): fresh seeds, fixed before running. Run 3's seeds (31-33, 700-719) were
# used to check the fix, so they no longer count.
SEEDS_RUN4 = [41, 42, 43]
COVERAGE_SEEDS_RUN4 = list(range(800, 820))

# Run 5 (after the run-4 fixes: windowed oscillator search, profile check, 2026-11-17):
# fresh seeds, fixed before running. Run 4's seeds (41-43, 800-819) were used to fix.
SEEDS_RUN5 = [51, 52, 53]
COVERAGE_SEEDS_RUN5 = list(range(1000, 1020))   # 900-939 were diagnosis seeds

# Run 6 (1.1: generous noise cap for built-in models, TESTPLAN "1.1"), 2026-11-25:
# fresh seeds, fixed before any 1.1 code or run. No bench record uses them (max seen 1179).
SEEDS_RUN6 = [61, 62, 63]
COVERAGE_SEEDS_RUN6 = list(range(1200, 1220))
