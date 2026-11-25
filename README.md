# equationnet

Type a differential equation and give it your data. equationnet trains six simple
models: three that use the equation and three data-only baselines. It ranks them
by their error on points they never saw, and gives you each model as a standalone
script. You can add models of your own to the comparison.

No code needed:

```
pip install equationnet
equationnet
```

That starts a guided session in the terminal. It asks for your data file, a CSV that
you can drag into the window. Then it asks which column you measured, which is time,
and which equation to test: a starter such as logistic growth or the heat equation,
or your own. It trains the six models and says in plain words whether the equation helped,
with the fitted constants. Then it can save the returned model, or all six, as
scripts, predict at new times, or try another equation on the same data. Arrow keys
and Enter choose (with answers piped in, it shows numbered menus instead); Esc or
Ctrl-C leaves.

From Python:

```python
import equationnet as en

result = en.fit("u_tt + c*u_t + k*u = 0", "spring.csv")   # examples/spring.csv: columns t, u
print(result.table())
result.best.save("best_model.py")    # a standalone script: numpy + scipy/torch only
```

```
equation: u_tt + c*u_t + k*u = 0
          target u(t); fit c, k
split: random, 32 points to train, 8 held out

rank model                              held-out error    time   fitted
-----------------------------------------------------------------------
1  = Smoothing spline (data only)              0.05551    0.7s   -
2  = PINN (neural network + equation)           0.0571    2.7s   c=0.3257, k=4.043
3  = Polynomial + equation                     0.05727    4.5s   c=0.3261, k=4.038
4  = Equation fit (classical)                  0.05754   11.6s   c=0.324+-0.015, k=4.029+-0.043
5  = Polynomial (data only)                    0.05869    0.0s   -
6    Neural network (data only)                 0.2913    1.5s   -

best: Equation fit (classical). Held-out error is the RMS error on data the models never saw.
=     5 models are within the noise of the lowest error on these 8 held-out points, so the data cannot tell them apart.
      The one the equation constrains most is returned; pick="lowest" would return Smoothing spline (data only).
+-: one standard error, if the equation is exactly right (a wrong or approximate equation can be off by more).
```

(`examples/spring.csv`: 40 noisy points of a damped spring, true c = 0.3, k = 4,
noise standard deviation 0.05. Recorded in `bench/stages/stage_d_results.json`;
`fit` also prints one progress line per model first. Times vary from run to run.)

When the data cannot pin a constant down, a line says so. For example, on five of
the six points of NIST's biochemical oxygen demand data (`u_t = k*(L - u)`, random
split):

```
not pinned down: L: poorly determined: set to half or one and a half times its value (243.8), the classical fit refits the data as well. If you know it, pass known={'L': ...}. ...
not pinned down: k: consistent with 0 (classical fit 0.2214 +- 0.12): the data cannot tell it from zero. If you know it, pass known={'k': ...}. ...
```

NIST certifies L = 213.8 +- 12.4 and k = 0.547 +- 0.105 from all six points with the
start value fixed at zero. equationnet also fits the start value, and with five
points both come out flagged: L poorly determined, k consistent with zero.

When the equation models that score the same, together with the classical fit,
give values more than 10% and more than two standard errors apart for a constant
the classical fit does pin down, a separate "models disagree" line says so.

## Why it exists

A physics-informed neural network (PINN) writes a known equation into a network's
training. There are good libraries for it (DeepXDE, PINA, NeuroDiffEq, NVIDIA
PhysicsNeMo, NeuralPDE.jl). equationnet asks a different question: given an
equation and some data, which simple model should you use?

A PINN is not always the answer. On several forward problems, PINNs did not
outperform the finite element method in solution time or accuracy in Grossmann et
al. (IMA J. Appl. Math. 89, 143 (2024), doi:10.1093/imamat/hxae011). The finite element method plus an optimizer beat them on
noisy inverse problems in Jekic et al. (2025 preprint, arXiv:2509.20191). So by
default equationnet trains the classical fit and data-only baselines next to the
equation-based networks, and lets held-out data rank them.

## The models

| name | in the table | what it is | uses the equation |
|---|---|---|---|
| `equation_fit` | Equation fit (classical) | SciPy solver + least squares on the unknowns (no network) | yes |
| `poly_physics` | Polynomial + equation | Chebyshev polynomial trained on data + equation | yes |
| `pinn` | PINN (neural network + equation) | small neural network trained on data + equation | yes |
| `poly` | Polynomial (data only) | polynomial, degree chosen by cross-validation | no |
| `spline` | Smoothing spline (data only) | smoothing spline, smoothness chosen automatically | no |
| `nn` | Neural network (data only) | neural network on the data only | no |

When a model cannot handle an equation, the table says why in words.

## Writing equations

| you type | meaning |
|---|---|
| `u_t`, `u_xx`, `u_xt` | partial derivatives of the target (one letter per variable) |
| `u'`, `u''` | derivatives when there is one variable |
| `^` or `**` | powers |
| `sin cos tan exp log sqrt tanh abs pi` | math (also `ln`, `log10`, `log2`, `asin`/`arcsin` and the other inverse and hyperbolic functions, `sec`, `csc`, `cot`, `erf`, `erfc`) |
| `f(u)` | an unknown function of one argument, learned from the data |
| any other name that is not a data column | a constant: fitted, unless you pass it in `known={...}` |

Functions that are not smooth or take two arguments (`sign`, `max`, `min`, `floor`,
`atan2` and similar) stop with a message instead of being learned.

Examples: `u_t = r*u*(1 - u/K)`, `u_t = D*u_xx`, `u_t + u*u_x = nu*u_xx`,
`u'' + c*u' + f(u) = 0`.

The data needs one column per variable plus the target (default `u`). Columns
other than the target are the variables, in order.

## Options

```python
en.fit(equation, data,
       target="u",               # target column
       known={"k": 4.0},         # constants you already know
       models=["equation_fit", "pinn"],   # default: all; can include your own (below)
       split="random",           # or "forecast": hold out the end in time
       holdout=0.2, seed=0,
       cv=1,                     # see "How models are scored"
       predict_at={"t": [12, 14]},   # where you will want predictions (e.g. the future)
       pick="simplest",          # or "lowest"; see "How models are scored"
       config={"pinn": {"steps": 1000}})
```

`predict_at` matters for forecasting. The PINN and the physics polynomial also
enforce the equation over the range spanned by the data and those locations
(never their values, which you don't have). The classical ODE fit tries to start
its solve at the earliest of these locations and the data, so that predictions there
run forwards (see "Limits in this version" for when it cannot).

`config` sets per-model options: `steps` and `lbfgs` for `pinn`, `poly_physics`
and `nn`; `collocation` for `pinn` and `poly_physics`; `degree` for
`poly_physics`. A setting the model does not use is an error, never silently
ignored.

Models run in a fixed order (polynomial, spline, classical fit, physics
polynomial, data-only network, PINN, then any of your own), and each prints its
result as it finishes.

## Your own model

Put your own model in the same table. Anything with `fit(X, y)` and `predict(X)`
that `copy.deepcopy` can copy works (tested with scikit-learn estimators and plain
classes), and so does a plain function `predict(X)` that needs no training:

```python
result = en.fit("u_tt + c*u_t + k*u = 0", "spring.csv",
                models=["all", my_model, en.custom(other_model, name="gp")])
```

`X` has one column per variable, in the data's column order (the target left out),
and `y` is the target. A fresh copy of your model is trained for every held-out
split, on the same points as the built-in models, and scored on the same held-out
points. Your object itself is not changed.

A model is named after its class, a function after its name, and a lambda or
another callable with no name `your_model_1`, `your_model_2`, ... by position. To name it yourself, use
`en.custom(model, name=...)`: letters, digits, `_` or `-`, starting with a letter
or `_`, at most 40 characters (the same rule applies to the automatic names). A name
may not repeat another model's, built-in names included. `result.save_scripts("scripts/")` saves every model's script,
yours included.

What equationnet does not know about your model, it says rather than guesses:

- **Ties.** On a tie with a built-in model, the built-in one is returned, because
  equationnet cannot see how much your model is constrained. In the tie order,
  yours come after every built-in model, and those you mark as using the equation
  come first among yours. The table says so when yours had the lowest error, and
  `pick="lowest"` returns it anyway.
- **Constants and standard errors.** Constants are shown if your model has a
  `fitted_params()` method, standard errors if it has `uncertainty()`; both return
  `{name: value}`. They are kept out of the "not pinned down" and "models
  disagree" checks, since your `c` need not be the equation's `c`.
- **Uses the equation?** Your model counts as data-only unless you say otherwise:
  `en.custom(model, uses_equation=True)`.
- **Noise level.** Your model's training error is never used as the noise level
  in the tie rule, and where a model of yours is involved the rule leans toward
  the built-in models both ways. When yours has the lowest held-out error, a
  built-in model counts as tied if its error is within what noise could give, with
  the noise level capped at an upper confidence bound from your model's held-out
  errors (97.5% for one model chosen beforehand, somewhat less for the lowest of
  several): 2.45 times their RMS on 5 points, 1.39 times on 24. So built-in models
  that all fit badly do not tie with a far better model of yours, but on a few
  held-out points a built-in model up to about 4 times worse can (5 points). A model
  of yours counts as tied with the lowest through the two-standard-error test,
  as any model does, or within a noise level capped at the lowest model's own
  held-out RMS error: up to about 1.6 times it on 5 points and 1.3 times on 24.
  Built-in models compared with a built-in lowest keep the rule below, unchanged
  from 0.0.1, so the `=` marks can skip a model of yours that is closer than tied
  built-in models. The table then says so.
- **Script.** The saved file for your model is a stub: its class, its
  parameters if it reports them (`get_params()`, as scikit-learn models do), and
  how it was trained and called. equationnet cannot write out code it did not
  write.

A model of yours that fails (raises, returns the wrong number of values, or returns
NaN) gets a `failed:` line in the table, and the other models still run.

## How models are scored

- **Default** (`cv=1`): one held-out split (`holdout`, default 20%). Models are
  trained on the other 80% and ranked by their error on the 20%.
- **Cross-validation** (`cv=5`, or `cv="auto"` for only under 50 points): 5 folds
  for `split="random"`, or forecast origins for `split="forecast"`. A forecast
  origin trains on everything before a cut and scores the next 20% of the distinct
  times. Earlier origins step back a whole window, so no point is scored twice.
  Models are ranked by the mean over folds, then every model is retrained on all
  your data. Every model is trained once per fold and once more on all the data,
  so it is off by default. `cv="auto"` uses 5 folds (random) or 3 forecast origins.
- **Ties** (`pick="simplest"`, the default): models whose mean squared held-out
  error is within two standard errors of the lowest one's are marked `=` as tied.
  So is a model whose error is no larger than measurement noise alone would give on
  that many held-out points (97.5% of the time). The noise level is the smallest
  training error of the built-in data-only models, which assume no equation; if
  none of them ran, this second test is off (models of your own: see "Your own
  model"). It has a known weakness, not fixed: on very small data a data-only model
  can pass almost exactly through its training points, so the noise level comes out
  well below the true noise and the test does not fire. In stage A run 5 that is
  behind four of the six picks more than 3x worse than the best, including the worst
  (see "How well it works"). The opposite also happens, also not fixed for the
  built-in models: when all the data-only models fit badly, their training error is
  their misfit, the noise level comes out far above the true noise, and models many
  times worse than the lowest are marked tied. The returned model is still the first
  tied one in the order below, so check the held-out errors, not only the `=` marks.
  Among the tied models, the one the equation
  constrains most is returned: the classical fit, then the physics polynomial, then
  the PINN, then the data-only models, then models of your own. The table says when
  this happened and what `pick="lowest"` would return.
- **Small held-out sets.** With a handful of held-out points no rule can always
  tell close models apart, and the returned model can be several times worse than
  the best one (see "How well it works"). Treat the ranking as a guide, and use
  `cv=5` or more data when it matters.

No data yet? `en.generate("u_t = D*u_xx", "pinn")` returns the training script
for one model.

Command line (`future_times.csv`: a CSV with the variable columns, here `t`):

```
equationnet fit "u_tt + c*u_t + k*u = 0" spring.csv --split forecast --out scripts/
equationnet fit "u_tt + c*u_t + k*u = 0" spring.csv --cv 5 --predict-at future_times.csv
equationnet script "u_t = D*u_xx" --model pinn > train_pinn.py
equationnet models
```

## How it works

For each model equationnet renders a complete Python script from the equation,
runs that script, and gives you the same text. The script you save is exactly the
one that was scored. (For a model of your own the saved file is a stub; see "Your
own model".) Running it trains the model again on the data you give it, so
its numbers can differ slightly from the table's.

## Limits in this version

- It needs data. Pure forward solves with no measurements are not scored yet.
- Standard errors (`+-` in the table) come from the classical fit only, or from a
  model of your own that reports them. The classical fit's assume the equation is
  exactly right, and an approximate equation can be off by more.
- The table warns "not pinned down" when a constant's standard error is infinite
  or more than half its value, or the constant is within two standard errors of
  zero. The classical ODE fit also moves each constant to half and to one and a
  half times its value and refits the rest; if it fits the data as well after
  either move (within the 95% chi-squared limit for one number), the constant is
  flagged too, however small its standard error. Read those lines before trusting
  a constant.
- The classical solver handles one variable, or one space variable plus time in
  the form `u_t = F(u, u_x, u_xx)` with fixed-value edges.
- Classical ODE (ordinary differential equation) fit: a start read off the data
  (constants of any sign, screened over 24 orders of magnitude), plus 128 random
  guesses from 0.01 to 100. Each is scored with one solve, and the best 6 that
  differ get a full fit. Each guess also needs the start values (u and its
  derivatives at the first time). They are first read off a smooth curve through
  all the data and off a small polynomial through the first few points, whichever
  fits better, and are then fitted along with the constants. Second-order and
  higher equations (oscillators) also get a search on the first quarter of the time
  range, extended step by step to all of it. With the profile check above this
  makes their fits slower: in stage A, 1.25 to 1.8 times as long as before (a Duffing
  oscillator with 48 points took 22 to 43 s). On evenly spaced data
  a fast oscillation can pass through the same points as a slow one. When fits tie
  like that, the smoothest is returned and the table prints a note.
- Classical PDE (partial differential equation) fit: method of lines on 81 grid
  points with fourth-order differences. The start profile and edge values come
  from the smoothed data near the first time and at the two x edges. A correction
  to each (at every data x up to 25 points, 4 points in time) is fitted along with
  the constants, and the start profile always meets the edge values. The fit then
  checks itself on a grid twice as fine, up to 321 points, and adds any change left
  to the `+-`. The edge values are known only where there is data, so after the
  last training time they are held at their last fitted values. Before the first
  training time the fit returns its start profile. Neither is a prediction from the
  equation. The table notes it when the held-out points or `predict_at` reach past
  the training times (a forecast split, for example); a later call to `predict`
  prints nothing. With `cv` above 1 the returned model is retrained on all the
  data, so no note prints, though each forecast fold was scored with held edges.
- When predictions are wanted before the first point the classical ODE fit trains
  on (a held-out first point, or `predict_at`), the solve normally starts there and
  runs forwards. If the data cannot place that earlier start (an equation that
  forgets its start quickly, like strong damping), the fit follows a guess and the
  table says so. If the start cannot be moved at all, the table says that the
  predictions before the first training point are solved backwards.
- Unknown functions take one argument. Classical fits treat them as cubics.
- No Gaussian process among the built-in models (you can pass one as your own
  model).
- Parsing uses SymPy, which evaluates text. equationnet refuses any character an
  equation does not need (quotes, dots after names, square or curly brackets, `#`,
  `;`, newlines) and huge exponents before SymPy sees the text. Even so, do not feed
  it equations from people you do not trust, e.g. in a web service.

## How well it works

From `TESTPLAN.md`, stage A run 5. The test used 14 problems written before any
run, scored against the exact solution on a dense grid:
- stiff, nonlinear, forced and unevenly spaced problems, four of them PDEs;
- 10 to 500 points, noise 0.1% to 20% of the signal's standard deviation;
- constants from 4e-4 to 1.2e5;
- two problems built to be nonidentifiable.

Each ran on 3 fresh seeds and both splits, with default settings: 84 runs. The
data are made by the same equation that is fitted. The problems themselves are
not new to the code: fixes after runs 1 to 4 were diagnosed on them (with other
seeds).

Bars for these numbers were written before the first run and never moved. Run 5
met all of them except the one on picks, which was missed and accepted in writing.
TESTPLAN.md has all five runs, with the misses of each.

- When the equation is exact, a model that uses it was truly best in 84 of 84 runs.
- The default pick was within 1.5x of the truly best model in 75 of 84 runs (89%;
  the bar was 90%). 6 picks were more than 3x worse; the bar allowed none.
- Five of those 6 came with the table's warning that the ranking rests on fewer
  than 5 held-out points: the two smallest problems, with 10 and 20 points. In four
  of them (the worst, 52x, a PINN chosen on 2 held-out points; 26x and 6.3x, where a
  data-only model was returned; and 4.4x) the classical fit was truly best and scored
  at about the noise, and the noise-floor tie described under "Ties" would have
  returned it had its noise level not come out too low. In the fifth (11.9x) the
  classical fit was returned, tied with a physics polynomial that was better still,
  both far below the noise.
- On the 72 runs with 5 or more held-out points, 69 picks (96%) were within 1.5x.
  The worst was 3.8x, on a problem built to be nonidentifiable, where the classical
  fit has more free numbers than the data need. It was returned by the tie order,
  with an error against the exact solution of about 0.4 of the noise level, while a
  PINN was closer still. That problem was over 3x in four of the five runs and is
  not fixed.
- On the 12 identifiable problems, the classical constants were within 3 standard
  errors of the truth in 71 of 72 fits. In the other, both constants were just
  outside (3.2 standard errors).
- Over 20 seeds for each of four problem families, each constant's truth fell
  within 1.96 standard errors in 90% to 95% of cases (the bar was 85% to 98%).
- "not pinned down" fired on every run of the two nonidentifiable problems and on
  none of the identifiable ones.
- The PINN failed on 2 of 6 Gompertz runs (`log(K/u)` once training takes u through
  zero). The table says so, and the other models still ran.

`bench/headroom.py` and `bench/FINDINGS.md` are the smaller development benchmark
the defaults were tuned on.

## Real data

`bench/realG/` holds five reference datasets from the US National Institute of
Standards and Technology (NIST): real measurements with certified constants. The
expected outcome of each was written down before running (TESTPLAN.md, stage G).
- Each set was fitted with NIST's model written as an ODE (10 runs). For Misra1a
  and BoxBOD this also frees the start value, which NIST fixes at zero.
- Every constant the data could not determine was flagged: Eckerle4's two, and
  BoxBOD's k and L. With 5 training points and the start value freed, BoxBOD's L is
  not pinned down to within half its value. NIST's fit of all 6 points, with the
  start fixed, gives it 6%. Every unflagged constant was within 3 standard errors of NIST's
  certified value, except Misra1a on the random split (A and k at 3.1 and 3.3
  standard errors).
- 2 of the 9 outcomes written down in advance were not met (3 in the first run,
  before the profile check flagged BoxBOD's L on the forecast as expected);
  TESTPLAN.md explains each.
- These sets have 6 to 35 points, so 1 to 7 are held out. A data-only model was
  picked in 6 of the 10 runs, so on data this small the ranking is a rough guide.

`bench/realG/` also holds one PDE test on real data: a week of hourly soil
temperature at 5, 10, 20, 50 and 100 cm from NOAA's US Climate Reference Network
(station Lincoln 11 SW, Nebraska, April 2025; `bench/realG/soil_lincoln_2025.csv`),
fitted with the heat equation `u_t = a*u_xx`, with the top and bottom sensors as the
edges. Six outcomes were written in TESTPLAN.md before the run. Apart from the range
for a, committed earlier, they were committed only with the results, so the record
cannot prove the order. Two were missed, both on the random split. The classical fit
gave a = 0.0096 +- 0.0003 m^2/day, below the 0.02 to 0.09 written in advance (a
range from general knowledge of soils, not a cited source). Its held-out error (0.23
degrees C) was 4.7 times the smoothing spline's (0.05).

After the run, the data turned out not to follow one value of a. Read off how the
daily swing shrinks and lags between two depths (`bench/realG/soil_harmonics.py`; a
rough method that assumes a uniform layer and a steady daily cycle), a is about
0.01 m^2/day between 5 and 10 cm and 0.18 to 0.33 between any two depths from 10 cm
down. The classical fit matched the top pair, where the daily swing is largest.
These data cannot say why the top differs. A misplaced sensor would have to be
about 20 cm off to explain it. A different top layer, heat carried by water, or
weather that is not a steady daily cycle could each contribute; none was tested. The `+-` (3.5%) did not show the trouble. Three other things in
the output did: the classical fit ranked fourth, outside the noise; the three
models that use the equation gave a from 0.0006 to 0.0096 on the same split; and the forecast split
gave a = 0.0117 +- 0.0004, 4.1 combined standard errors from the random split's.

On the forecast split (the last 1.4 days held out) the outcomes written in advance
were met, including the trouble: the classical fit was best (1.8 degrees C held-out
error, against 3.0 for the PINN), but its error at 5 cm was 3.5 degrees because it
held the top edge at its last training value, and it printed no note saying so. The
note was added after the run (see "Limits in this version").

`bench/real/` holds six measured time series from the data sets of Lomen and
Lovelock's ODE textbook. They were run on an earlier version of equationnet;
`bench/real/FINDINGS_REAL.md` has the results, which predate the current tie rule
and fits. One lesson from them still holds: error bars assume the equation is
exact. There, the fitted room temperature for coffee cooling came out well above
the stated one. A likely reason is that Newton's law of cooling is only
approximately right for a cup of coffee, which no error bar computed from noise
can see.

## Status

1.0.0. New since 0.0.1: your own model in the comparison ("Your own model", tested
in stage J of `TESTPLAN.md`). The built-in models are unchanged, and so is the tie
rule for runs without models of your own, so the results above still apply. The arguments of `fit()` and the layout of the table
are kept stable within 1.x.

## License

The code and documentation are under the MIT license (`LICENSE`). The data tables
in `bench/real/datasets.py` and `bench/realG/datasets_g.py` are third-party
measurements, not covered by that license. They are reproduced as published, with
credit to the sources named in each file; neither source page states license
terms. If you hold rights in any of them and object, open an issue and they will
be removed.

`bench/realG/soil_lincoln_2025.csv` is derived from NOAA's US Climate Reference
Network hourly data (NOAA National Centers for Environmental Information), a US
government work in the public domain; `bench/realG/soil.py` rebuilds it from the
published file.
