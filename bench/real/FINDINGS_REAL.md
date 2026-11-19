# Real data: findings (2026-11-16)

**Question.** Does equationnet hold up on real measurements, where the noise is not
Gaussian, the data are few, and the equation may be only roughly right?

**Data.** Six real time series with a known governing equation, from the data sets
of Lomen and Lovelock, "Differential Equations: Graphics, Models, Data" (Wiley,
1999), https://archive.math.arizona.edu/dsl/datall.htm. Values, original sources
and what is known independently are in `datasets.py` (transcription checked
against the pages: same counts and column sums). Run: `python bench/real/run_real.py`.
Default settings (one 20% split, tie rule), both splits, all six models. There is
no exact solution, so models are judged by held-out real points and by fitted
constants against what is known independently.

| data | points | equation | known independently |
|---|---|---|---|
| yeast (Carlson 1913) | 19 | `u_t = r*u*(1 - u/K)` | published logistic fits: K about 665 |
| coffee cooling | 11 | `u_t = -k*(u - A)` and with A = 24 | room at 24 C |
| series RLC circuit | 9 | `u_tt + c*u_t + k*u = 0` | from R, L, C: c = 4167, k = 476,190, slow decay rate 118 /s |
| overdamped pendulum | 27 | `u_tt + c*u_t + k*u = 0` | nothing |
| theophylline in blood | 11 | `u_t = -k*u` | nothing checked |
| dye in calf plasma | 17 | `u_t = -k*u` and `u_t = -f(u)` | usually two-compartment, so `-k*u` should be wrong |

## Found and fixed: physics models started every constant at 1

The PINN and the physics polynomial started each constant at 1 and only refined
locally. Every synthetic benchmark constant was of order 1, so this never showed.
On real data it failed at once:

- Yeast: K = 1.6, r = -0.0003 (published K about 665). Forecast error 1039 and 1480
  against the classical fit's 3.1.
- Coffee: room temperature 1.1 to 1.8 C. RLC circuit: k = 0.44 (true 476,190).

Fix: before refining, screen 4096 values per constant over 12 decades and both signs
on the fixed warm-up curve (cheap: no training), then 4096 within one decade of the
best. The classical ODE fit's regression start got the same screen. After it, yeast
gives K = 657 to 673 and r = 0.53 to 0.56 from all three equation models. Synthetic
benchmark (run 7): unchanged except where the physics models already failed
(missing_term forecasts, never picked).

## Results after the fix

Held-out error on real points; best model in bold.

| data / split | classical | PINN | poly+eq | best data-only | picked |
|---|---|---|---|---|---|
| yeast / random | 2.96 | 2.96 | 2.95 | **2.39** (NN) | classical (4 tied) |
| yeast / forecast | **3.11** | 8.17 | 5.73 | 13.2 (NN) | classical |
| RLC / random | **0.00047** | 0.0014 | 0.00082 | 0.0013 (NN) | classical (4 tied) |
| RLC / forecast | **0.00021** | 0.0029 | 0.00027 | 0.0015 (NN) | classical |
| coffee, A unknown / random | 0.365 | **0.165** | 0.167 | 0.350 (spline) | poly+eq (2 tied) |
| coffee, A unknown / forecast | 0.202 | 0.358 | 0.201 | **0.195** (NN) | classical (6 tied) |
| coffee, A = 24 / random | **0.268** | 0.286 | 0.286 | 0.350 (spline) | classical |
| coffee, A = 24 / forecast | 0.183 | **0.181** | 0.184 | 0.195 (NN) | classical (6 tied) |
| theophylline / random | 0.148 | 0.134 | 0.136 | **0.052** (spline) | spline |
| theophylline / forecast | 0.043 | **0.016** | 0.020 | 0.393 (NN) | PINN |
| overdamped pendulum / random | 0.021 | 0.044 | 0.030 | **0.018** (poly) | poly (3 tied) |
| overdamped pendulum / forecast | 0.032 | 0.071 | **0.028** | 0.121 (NN) | classical (2 tied) |
| calf dye, `-k*u` / random | 0.298 | 0.260 | 0.269 | **0.025** (NN) | NN |
| calf dye, `-k*u` / forecast | 0.183 | 0.158 | 0.164 | **0.015** (NN) | NN |
| calf dye, `-f(u)` / random | 0.041 | 0.027 | 0.087 | **0.025** (NN) | NN |
| calf dye, `-f(u)` / forecast | 0.084 | 0.033 | 0.107 | **0.015** (NN) | NN |

Held-out sets are 2 to 6 points, so single rankings are noisy; "tied" is the tie
rule's count.

## What it says

1. **When the equation is right, it wins forecasts on real data by a wide margin.**
   Yeast 4x, RLC 7x, theophylline 25x better than the best data-only model. On
   interpolation (random split) data-only models are often as good or better.
2. **The fitted constants are right where the data determine them.** Yeast: K = 657
   to 673, r = 0.53 to 0.56 (published K about 665). RLC: the slow decay rate is
   108 to 112 /s from the classical fit and the physics polynomial, 132 to 134 /s
   from the PINN, against 118 /s from the component values. (The PINN's constants
   also imply a slowly growing mode, +3 to +14 /s, that is physically wrong but too
   slow to show within 0.04 s.)
3. **Often the data do not determine every constant, and equationnet does not say
   so.** This is the main gap found.
   - RLC: c ranges from 117 to 3226 across models (true 4167). The decay looks like
     a single exponential, which fixes only one combination of c and k.
   - Coffee: room temperature comes out anywhere from 0.05 to 32 C (stated: 24).
     Ten minutes of cooling cannot separate A from k. With A = 24 given, every model
     agrees on k = 0.0427 to 0.0429 /min.
   - On the coffee random split the tie rule returned the physics polynomial with a
     room at 0.8 C, with no warning.
4. **A wrong equation is caught by the ranking.** For the dye, one exponential is
   10x worse on held-out points than the data-only network on both splits. Letting
   the decay law be learned (`-f(u)`) brings the PINN to within 1.1x to 2.3x of it.
   For theophylline the one-compartment law is beaten by a spline in between the
   points (0.05 against 0.13) but wins the forecast 25x. Approximately right
   equations forecast well even when they interpolate worse.
5. **One oddity of the paired tie rule.** On the overdamped pendulum (random split,
   5 points) the data-only network (0.047) was tied with the lowest but the
   classical fit (0.021) was not: a model whose extra error is consistent counts as
   worse than one whose error swings widely. Statistically correct, but surprising
   in a table.

## Added after this test: how well each constant is determined

The classical fit now reports a standard error for each constant (from its fit's
Jacobian, by SVD, so a direction the data do not constrain gives an infinite error)
and the table warns in words ("not pinned down") when a constant has an infinite
error, an error over half its value, or when the equation models the held-out data
cannot tell apart (the tied set, plus the classical fit) disagree by more than 10%
and more than 2 standard errors. A first version compared every equation model
within 2x of the best error; on the synthetic benchmark that raised one false alarm
(heat, where a badly fitting physics polynomial said D = 0.001 against a classical
0.097 +- 0.002), so it now uses the tied set. On the synthetic benchmark (run 8) no
constant is flagged.

On synthetic data the errors are honest: the truth fell within 1.96 standard
errors in 28 of 30 heat fits, 56 to 57 of 60 logistic fits and 54 to 55 of 60
oscillator fits (the oscillator's errors run 15 to 30% small). Checking this found
one more classical-fit failure: on noisy logistic data the fit's starting value
for u came from a smooth curve that dipped below zero, where the logistic equation
runs away, so 3 of 60 fits collapsed to K = 0.2. It now falls back to starting
values from the data.

On the real data (all 16 runs):

| data | warned about | not warned |
|---|---|---|
| yeast | nothing | K 663 +- 2 / 660 +- 4, r 0.547 +- 0.007 |
| RLC circuit | c and k, both splits (classical errors infinite; models 1116 to 3226 for c) | |
| coffee, A unknown | A both splits (models 0.05 to 32), k on the random split | |
| coffee, A = 24 | nothing | k 0.0429 +- 0.0003 |
| overdamped pendulum | c and k, both splits | |
| theophylline | nothing | k 0.171 +- 0.003 |
| calf dye, `-k*u` | nothing | k 0.248 +- 0.007, but the equation is wrong (the ranking shows it) |

The standard errors assume the equation is exactly right, and the table says so.
On coffee they put room temperature at 32 +- 3 C against the stated 24: profiling
it (refitting with the room temperature held fixed) shows the training points really do prefer 32, while the held-out
points prefer 0. Newton's law is only approximately right for cooling coffee, and
noise-based errors cannot see that. The models-disagree check is what caught it.

## Next

- Standard errors for the PINN and the physics polynomial (now they are only
  compared against the classical fit and each other).
- More real sets with PDEs; these are all ODEs.
