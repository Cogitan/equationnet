# Stage J6: fresh-user review of "your own model" (2026-11-24)

Reviewer: a separate Claude agent (Anthropic), told to act as a scientist new to the
package, allowed to read only README.md, using the installed package. It made its own
data (60 points, u = exp(-0.25 t) cos(1.8 t) + N(0, 0.03), seed 42) and passed a
scikit-learn GradientBoostingRegressor and a GaussianProcessRegressor (named "my_GP")
with `models=["all", gbr, en.custom(gp, name="my_GP")]`. Run on the code at a1e9ab3.

## Its answers (checked against its table)

- Q1: an equation model had the lowest error (PINN 0.02873, physics polynomial
  0.02875, classical 0.0288); five models tied, including my_GP; returned: the
  classical fit, by the tie order; pick="lowest" would return the PINN. Correct.
- Q2: for its models the table shows rank, tie mark, held-out error and time; no
  constants, no standard errors, a stub script. Correct.
- Q3: c = 0.500 +- 0.010, k = 3.263 +- 0.021 (classical); true c = 0.5, k = 3.3025.
  No warning lines, which it judged right.

J6 bar: answers correctly. MET, with the confusions below.

## Confusions it reported, and what was done

| # | confusion | disposition |
|---|---|---|
| 1 | naming rules undocumented; error did not say which character | fixed: README states the rule; the error names the bad character(s) |
| 2 | default name of an unnamed model undocumented | fixed: README |
| 3 | `result.save_scripts()` not in the README | fixed: README |
| 4 | "six models" vs eight rows | accepted: six built-in models is still true; the "Your own model" section says yours are added |
| 5 | "Gaussian processes are not in yet" read as a restriction | fixed: "No Gaussian process among the built-in models (you can pass one as your own model)" |
| 6 | the stub records almost nothing | fixed: the stub records the parameters from `get_params()` when the model has it |
| 7 | the stub imports from a private module (`sklearn.ensemble._gb`) | fixed: trailing private module names are dropped |
| 8 | the `yours:` note named `uncertainty()` but not `fitted_params()` | fixed: names both (a later fix, after the writing preflight's V3/V4, dropped the tie outcome from this note; the tie lines state it) |
| 9 | the tie line does not say which of the two tie tests applied | accepted: pre-existing 0.0.1 wording, documented in "How models are scored" |
| 10 | `best:` names a model ranked 3rd | accepted: pre-existing; the next two lines explain it, and the table layout is kept stable |
| 11 | number formatting (0.0288, 0.5+-0.01) | accepted: pre-existing 4-significant-figure / 2-figure format |
| 12 | `print(result.best)` printed the whole script | fixed: the script is left out of the repr |
| 13 | `constant_notes` is a method beside attributes | accepted: pre-existing API |
| 14 | where a model of yours falls in the tie order | fixed: README says yours come after every built-in model |

The fixes change messages, the stub, the repr and the README only; no score, tie or
pick. Stage J2-J5 ran on a1e9ab3's scoring code, which they do not touch.
