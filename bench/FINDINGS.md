# Headroom test: findings (run 6, 2026-11-16)

**Question.** If you type an equation, does anything beat the plain classical fit
(SciPy solver + least squares)? And does "train them all, pick the best on
held-out data" actually pick well?

**Setup.** `bench/headroom.py`, default model settings, CPU. 3 problems x 2 splits x
3 seeds = 18 runs, 6 models each. Every run holds out the same 20% of points
(random, or the last 20% in time). Models are scored against the **exact
solution** in two ways: on the held-out points, and on a dense grid (the whole
data range for random splits, the whole forecast window for forecasts). The dense
grid is the better judge of "truly best": on the oscillator the held-out set is
only 4 points.

Raw rows: `results_run6_s{0,1,2}.jsonl` (current: same models as run 5, plus every
model's per-point held-out errors and the tie-rule pick). Kept as the record:
`results_run5_s*.jsonl` (after the speed work; model errors identical to run 6),
`results_run4_ode_s*.jsonl`, `results_run3_s*.jsonl`, `results_run2.jsonl`,
`results_run1.jsonl`.
Report: `python bench/headroom.py --report results_run6_s0.jsonl results_run6_s1.jsonl results_run6_s2.jsonl`.

| problem | data | unknowns |
|---|---|---|
| oscillator | `u_tt + c*u_t + k*u = 0`, 20 points over ~3 periods, noise 0.05 | c, k |
| heat | `u_t = D*u_xx`, 11 x 11 grid, noise 0.02 | D |
| missing_term | `u_tt + c*u_t + f(u) = 0`, true f = 3 tanh(2u) (not a polynomial), 30 points, noise 0.05 | c and the function f |

## Results (mean error against the exact solution on the dense grid, 3 seeds)

| problem / split | classical | PINN | poly+eq | NN (data) | spline (data) | truly best (wins of 3) |
|---|---|---|---|---|---|---|
| oscillator / random | **0.015** | 0.022 | 0.025 | 0.072 | 0.287 | classical 3 |
| oscillator / forecast | **0.014** | 0.017 | 0.019 | 0.475 | 2.12 | classical 2, poly+eq 1 |
| heat / random | **0.008** | 0.010 | 0.029 | 0.026 | 0.017 | classical 2, polynomial (data) 1 |
| heat / forecast | **0.007** | 0.014 | 0.094 | 0.048 | 0.027 | classical 3 |
| missing_term / random | 0.037 | **0.031** | 0.031 | 0.059 | 0.048 | PINN 2, poly+eq 1 |
| missing_term / forecast | **0.072** | 0.581 | 0.588 | 0.596 | 3.86 | classical 3 |

Seconds per benchmark run (small-data runs include 5 folds plus the refit, so 6
fits), run 3/4 -> run 5:

| problem | classical | PINN | poly+eq | NN (data) |
|---|---|---|---|---|
| oscillator | 28 -> 26 | 161 -> 51 | 198 -> 18 | 22 -> 17 |
| heat | 35 -> 3 | 62 -> 22 | 134 -> 4 | 7 -> 5 |
| missing_term | 90 -> 90 | 164 -> 68 | 206 -> 45 | 21 -> 20 |

All models together: 376 s -> 123 s per run. Spline and polynomial: under 0.1 s.

## What it says

1. **Writing the equation in is worth it.** An equation-based model was truly best
   in 17 of 18 runs; the exception is a 1% tie on the 121-point heat grid (data-only
   polynomial 0.0094, PINN 0.0095). On forecasts, the best data-only model is 4x to
   34x worse.
2. **The classical fit is the right default, not the whole answer.** Truly best in
   13 of 18, including 8 of 9 forecasts. The PINN or physics polynomial won
   interpolation with an unknown function (missing_term/random, where the classical
   fit's cubic f costs 1.2x) and one oscillator forecast.
3. **Learned unknown functions do not forecast.** In missing_term/forecast every
   network-based model was as bad as the data-only network (about 0.59) while the
   classical fit, with f as a cubic, got 0.07. Neural f is pinned only where data is.
4. **Picking on held-out data mostly works; a tie rule makes its misses small;
   cross-validation did not help.** Against the dense grid:

   | rule | right | worst miss | mean cost over 18 runs |
   |---|---|---|---|
   | lowest error, one 20% split | 13 of 18 | 3.09x | 1.130x |
   | **tie rule (default): one split, 1 standard error** | 13 of 18 | **1.30x** | **1.044x** |
   | tie rule, 0.5 standard errors | 13 of 18 | 3.09x | 1.134x |
   | tie rule, 2 standard errors | 13 of 18 | 1.30x | 1.054x |
   | lowest error, 5-fold cross-validation | 12 of 18 | 3.09x | 1.155x |
   | tie rule on all 5 folds pooled | 12 of 18 | 3.09x | 1.150x |

   The tie rule: models whose extra squared error on the same held-out points is
   within one standard error of that difference (paired) are tied with the lowest,
   and among them the one with the fewest free numbers is returned (classical, then
   physics polynomial, then PINN, then the data-only models). One standard error is
   the textbook rule and was chosen before this run; 0.5 and 2 are shown only to say
   how sensitive it is. It removes the 3.1x oscillator miss and costs 1.16x to
   1.30x where the classical fit is preferred over an equation model that was in fact
   better (missing_term, where its cubic f is the weakness, and one oscillator
   forecast). Pooled over cross-validation folds it does not help: on 16 pooled
   points the physics polynomial's lower error passed as real. So it is the
   default with one split, and `pick="lowest"` turns it off.

   Before the tie rule, every miss was a near tie in held-out score. The
   largest (oscillator seed 0, 3.1x on the dense grid) was first blamed on a failed
   classical fit in one fold; that fit is fixed (fold error 0.52 to 0.04), and the
   miss stays: classical scores 0.042, the physics polynomial 0.038, while their true
   errors are 0.009 and 0.028. With 0.05 noise on 16 points, held-out error cannot
   see a difference that small. So one split stays the default; cross-validation is
   an option (`cv=5` or `cv="auto"`).
5. **Held-out data cannot catch an alias.** On evenly spaced samples a spring at
   k = 96 passes through the same points as the true k = 4. The old classical fit
   returned that answer silently in one fold, and it scored fine there (0.069, a
   normal fold), while its true error was 0.60. See the run 4 change below.

Scissors check: the headroom is real but narrow. Networks win interpolation when
part of the equation is unknown; the classical solver wins forecasts and small
data. The honest pitch is "type the equation, get the best model and its code",
with the classical fit usually winning. Never "PINNs beat solvers".

## Change in run 7 (constant screen, 2026-11-16)

Real data (`bench/real/FINDINGS_REAL.md`) showed the PINN and physics polynomial
started every constant at 1 and refined locally, so a constant far from 1 (yeast
carrying capacity 665) was never reached; every constant here is of order 1, which
hid it. They now screen 4096 values per constant over 12 decades and both signs on
the warm-up curve, then 4096 within a decade of the best; the classical ODE fit's
regression start got the same screen. On this benchmark (`results_run7_s*.jsonl`)
nothing changed except where the physics models already failed: missing_term
forecasts, PINN 0.58 to 0.70 and physics polynomial 0.59 to 0.62 (the classical
fit, 0.07, is picked there). The default pick: 14 of 18 right, worst miss 1.52x,
mean cost 1.061x (run 6: 13, 1.30x, 1.044x), from small shifts in near ties.

## Change in run 6 (tie rule, 2026-11-16)

`fit()` keeps every model's residual on every held-out point, marks the models tied
with the lowest error (`=` in the table) and returns the simplest of them; the
table says so in words and names what `pick="lowest"` would return. Results above
(finding 4). The README's own example shows why it matters: on 40 noisy spring
points, five of six models were within the noise on 8 held-out points.

## Changes in run 5 (speed, 2026-11-16)

Profiling showed nearly all the physics models' time was the Adam steps (6 to 10 ms
each), and the classical PDE fit's time was the solver plus a guess (D = 10) that
wandered to D = 900.

- **Physics polynomial: derivatives as fixed matrices.** The polynomial is linear in
  its coefficients, so its values and derivatives at the fixed equation points are
  fixed matrices times the coefficients (Chebyshev derivative recurrence, checked
  against autograd to 1e-15). Built once, each step is a few matrix products. Same
  answers, 4x to 30x faster.
- **PINN: 1000 Adam steps instead of 3000.** Tested 500, 1000, 2000, 3000 on all
  three problems, both splits, 3 seeds: accuracy was the same at every count (the
  L-BFGS finish does the work; 500 was marginally best). 1000 leaves margin for
  harder problems.
- **Classical PDE fit: 41 grid points and tolerance 1e-5** (was 81 and 1e-6): 8x
  faster per solve, solver error 5e-4 against a fit error of about 0.008. And the
  constants are screened with one solve per guess (7 values, 0.01 to 10), then
  fitted from the best only. Fitted D unchanged to the third decimal.
- **Models run fastest first** and each prints as it finishes.
- Found on the way: `generate()` silently ignored settings passed in the `fit()`
  form (`{"pinn": {"steps": 500}}`), which made the first step-count test
  meaningless. Both forms work now, and a setting a model does not use is an error.

## Change in run 4 (classical ODE fit, 2026-11-16)

Checked whether the ODE fit's tiny finite-difference step (~1e-8 against solver
tolerance 1e-8) was behind its bad folds. It was not. Every bad fit (5 of 30
fold and final fits on the two ODE problems) was a wrong starting guess:

- On sparse folds (12 to 19 points over 3 swings) the smooth curve the regression
  start is built on cannot follow the oscillation, so that guess was junk in every
  fold, and only 4 random guesses were left to find the right frequency.
- Fix: 128 random guesses (constants 0.01 to 100; each unknown function starts as
  a random straight line) are each scored with one solve after a one-step refit of
  the starting values, and the 6 best that differ get a full fit.
- That search finds aliases too (k = 99 fits seed 0 as well as k = 4). Among fits
  that tie within the noise, the one with the fewest turning points is returned,
  and the results table prints a note when a faster answer tied (3 of 6 benchmark
  runs on the oscillator and missing_term print it).
- Result on all 30 fold and final fits: 0 bad (dense error over 0.2) against 5
  before. Classical missing_term/random dense error 0.061 to 0.037; the oscillator
  and forecasts unchanged. Time per fit: oscillator 4 to 6 s, missing_term 7 to
  23 s mean.
- The step (`DIFF_STEP = 1e-4` against SciPy's default) changed little: better on
  2 of 30 fits (0.050 against 0.080, 0.043 against 0.070), up to 2x slower on
  missing_term. Kept for the reason it was suspected, but it was not the problem.

## Changes in run 3 (after run 2)

- **Classical PDE fits its start profile and edges** instead of reading them off
  the data. They start from the smoothed data; a smooth correction (5 points in x,
  4 in time) is fitted along with the constants. Heat/random dense error went
  from 0.017 to 0.008 (seed 1, which had lost four neighbouring start points: 0.033
  to 0.010, D 0.113 to 0.106). Forecasts unchanged (0.0067 to 0.0069).
  What it took, all found along the way:
  - Nine free start values instead of a correction fitted the noise (worse than
    before on complete data); a correction at 9 points also overfits. 5 points is
    the setting that helped everywhere it was tried.
  - The fit's default finite-difference step (~1e-8) was far below the PDE
    solver's error (~1e-6), so its gradients were noise. Now `DIFF_STEP = 1e-3`.
  - Even with a good step, SciPy's least squares stalled on the joint problem
    (constants + corrections) at 3x the achievable error. The corrections are now
    fitted by a direct linear solve (exact when the equation is linear in u, one
    Gauss-Newton step otherwise), alternating with the constants, 3 rounds.
  - The solver's Jacobian is now declared tridiagonal (each point sees its
    neighbours).
- **Cross-validation option** (`cv`), with every model retrained on all the data
  afterwards, and **`predict_at`** (where predictions will be wanted: physics
  models enforce the equation there). The benchmark now gives fit() only the 80%
  for small data and passes the held-out locations as `predict_at`, so the pick
  never sees the evaluation points.
- **Dense-grid scoring** against the exact solution in the benchmark.

## Three fixes made between run 1 and run 2 (all found by run 1)

- **Equation enforced only inside the training window** (physics models): in
  forecast mode the future got no help from the equation. Now it is enforced over
  every location where predictions are wanted (locations only, never held-out
  values). PINN oscillator forecast went from 0.099 to 0.017.
- **Physics polynomial warm-up threaded the noise** (degree 24 on ~16 points):
  now a degree-weighted smoothing start. Its oscillator error went from 22 to 0.026.
- **Classical PDE fed noisy start and edge values into every step**: now
  smoothed. Heat/forecast went from 0.011 to 0.008.

Also found while building: the polynomial baseline's cross-validation scored
extrapolation (it now keeps edge points in every fold), and a strong data-only
baseline (smoothing spline) was missing, so it was added.

## Known weaknesses (next steps, in order)

1. **Speed, what is left.** The classical ODE fit with an unknown function
   (missing_term) is now the slowest, about 15 s per fit (128-guess screen, 6 full
   fits with a cubic f); the PINN is 8 to 14 s per fit. The data-only network still
   runs 3000 steps, untested at fewer.
2. **Small-data picking is a noise floor.** The tie rule turned a 3.1x worst
   case into 1.3x but did not raise the hit count (13 of 18), and with
   cross-validation it does not help. The held-out sets here are 3 to 5 points;
   only more data fixes that.
3. **The ODE search assumes positive constants of size 0.01 to 100.** Negative or
   far larger constants rely on the local fit getting there. The PDE fit has no such
   screen yet (3 fixed guesses for D).
4. **Physics polynomial in 2D** is slow and weak (D about 0.055 on heat); drop it
   from the default set in 2D unless it improves.
5. Not covered yet: Gaussian processes, forward solves with no data, real
   (non-synthetic) data.
