"""Headroom test: does writing the equation into a model ever beat the plain classical fit?

Three problems, two held-out splits, three seeds, default settings (what a user gets).
Every run holds out the same 20% of points (random, or the last 20% in time) and scores
each model there against the clean truth (what actually matters).

How the 80% is handed to equationnet follows its defaults:
  small data (under 50 points): fit() gets only the 80%, cross-validates on it to rank
      the models, retrains each on all of it, and is told where predictions will be
      wanted (predict_at). The pick never sees the held-out 20%.
  larger data (heat, 121 points): fit() gets everything and makes the identical 20%
      split itself (one split, nothing to cross-validate), so its held-out set is ours.

Rows append to a .jsonl file, so a stopped run resumes where it left off.

    python bench/headroom.py --out results.jsonl                 # run
    python bench/headroom.py --report results_run2.jsonl results.jsonl   # later files override
"""

import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import equationnet as en  # noqa: E402
from equationnet.fit import SMALL_DATA, _split  # noqa: E402


def oscillator(seed):
    """Damped spring, 20 noisy points over ~3 periods. Unknown c, k."""
    c, k = 0.3, 4.0
    t = np.linspace(0, 10, 20)
    sol = solve_ivp(lambda s_, s: [s[1], -c * s[1] - k * s[0]], (0, 10), [1.0, 0.0],
                    dense_output=True, rtol=1e-10, atol=1e-12)
    truth = lambda X: sol.sol(X[:, 0])[0]
    clean = truth(t[:, None])
    noisy = clean + 0.05 * np.random.default_rng(seed).standard_normal(len(t))
    return dict(equation="u_tt + c*u_t + k*u = 0", data={"t": t, "u": noisy}, clean=clean,
                truth=truth, true_params={"c": c, "k": k})


def heat(seed):
    """Heat equation on a coarse 11 x 11 grid, 2% noise. Unknown D."""
    D = 0.1
    x, t = np.linspace(0, 1, 11), np.linspace(0, 1, 11)
    XX, TT = np.meshgrid(x, t)
    truth = lambda X: (np.exp(-D * np.pi ** 2 * X[:, 1]) * np.sin(np.pi * X[:, 0])
                       + 0.5 * np.exp(-4 * D * np.pi ** 2 * X[:, 1]) * np.sin(2 * np.pi * X[:, 0]))
    U = truth(np.column_stack([XX.ravel(), TT.ravel()]))
    noisy = U + 0.02 * np.random.default_rng(seed).standard_normal(U.shape)
    return dict(equation="u_t = D*u_xx", data={"x": XX.ravel(), "t": TT.ravel(), "u": noisy},
                clean=U, truth=truth, true_params={"D": D})


def missing_term(seed):
    """Spring with an unknown restoring force f(u) = 3 tanh(2u) (not a polynomial). Unknown c, f."""
    c = 0.3
    f = lambda z: 3 * np.tanh(2 * z)
    t = np.linspace(0, 10, 30)
    sol = solve_ivp(lambda s_, s: [s[1], -c * s[1] - f(s[0])], (0, 10), [1.5, 0.0],
                    dense_output=True, rtol=1e-10, atol=1e-12)
    truth = lambda X: sol.sol(X[:, 0])[0]
    clean = truth(t[:, None])
    noisy = clean + 0.05 * np.random.default_rng(seed).standard_normal(len(t))
    return dict(equation="u_tt + c*u_t + f(u) = 0", data={"t": t, "u": noisy}, clean=clean,
                truth=truth, true_params={"c": c}, true_f=f)


PROBLEMS = {"oscillator": oscillator, "heat": heat, "missing_term": missing_term}


def dense_points(X, indep, train, split):
    """Where the 'dense' error is measured against the exact solution: a fine grid over
    the whole data range (random split) or over the held-out future (forecast split)."""
    lo, hi = X.min(0), X.max(0)
    ti = indep.index("t")
    if split == "forecast":
        lo = lo.copy()
        lo[ti] = X[train, ti].max()
    n = 400 if X.shape[1] == 1 else 41
    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi)]
    return np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])


def done_keys(out):
    if not out.exists():
        return set()
    return {(r["problem"], r["split"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())}


def run(seeds, problems, models, out):
    done = done_keys(out)
    for seed in seeds:
        for split in ("random", "forecast"):
            for pname in problems:
                if (pname, split, seed) in done:
                    continue
                p = PROBLEMS[pname](seed)
                print(f"\n== {pname} / {split} / seed {seed}", flush=True)
                t0 = time.perf_counter()
                indep = [k for k in p["data"] if k != "u"]
                X = np.column_stack([p["data"][k] for k in indep])
                train, test = _split(X, p["data"]["u"], indep, split, 0.2, seed)
                if train.sum() < SMALL_DATA:
                    data = {k: np.asarray(v)[train] for k, v in p["data"].items()}
                    res = en.fit(p["equation"], data, split=split, seed=seed, models=models,
                                 cv="auto", predict_at=X[test])
                else:
                    res = en.fit(p["equation"], p["data"], split=split, seed=seed, models=models, cv="auto")
                    assert res.folds == 1 and res.n_holdout == test.sum()
                row = {"problem": pname, "split": split, "seed": seed, "folds": res.folds, "models": {},
                       "pick": res.best.name if res.ranked else None,
                       "tied": [c.name for c in res.tied]}
                for name, c in res.candidates.items():
                    m = {"status": c.status, "seconds": round(c.seconds, 2)}
                    if c.ok:
                        pred = c.predict(X[test])
                        m["score"] = c.holdout_rmse            # what the pick uses
                        m["fold_scores"] = c.fold_rmse
                        m["fold_errors"] = [e.tolist() for e in c.fold_errors]
                        m["holdout_true"] = float(np.sqrt(np.mean((pred - p["clean"][test]) ** 2)))
                        Q = dense_points(X, indep, train, split)
                        m["dense_true"] = float(np.sqrt(np.mean((c.predict(Q) - p["truth"](Q)) ** 2)))
                        m["params"] = c.params
                        if "true_f" in p and c.uses_equation:
                            z = np.linspace(-1.2, 1.2, 25)
                            m["f_rmse"] = float(np.sqrt(np.mean((c.learned_function("f", z) - p["true_f"](z)) ** 2)))
                    row["models"][name] = m
                row["seconds"] = round(time.perf_counter() - t0, 1)
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(res.table(), flush=True)


def load(files):
    """Rows keyed by (problem, split, seed); a later file's models override an earlier one's."""
    rows = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            key = (r["problem"], r["split"], r["seed"])
            if key in rows:
                rows[key]["models"].update(r["models"])
                rows[key]["folds"] = r.get("folds", 1)
            else:
                rows[key] = r
    for r in rows.values():                     # run 2 rows call the pick score holdout_noisy
        for m in r["models"].values():
            if "score" not in m and "holdout_noisy" in m:
                m["score"] = m["holdout_noisy"]
    return list(rows.values())


def pick_report(rows):
    """How often the pick (lowest score) was the truly best model, by the new rule
    (cross-validation where used) and by the old one (a single 20% split = fold 1)."""
    for truth_key, what in (("holdout_true", "on the held-out points"),
                            ("dense_true", "on a dense grid (whole range, or the whole forecast window)")):
        if not all(truth_key in m for r in rows for m in r["models"].values() if "holdout_true" in m):
            continue
        print(f"Pick accuracy (truly best = lowest error against the exact solution {what}):")
        _picks(rows, truth_key)


SIMPLEST_FIRST = ["equation_fit", "poly_physics", "pinn", "poly", "spline", "nn"]


def tie_pick(ok, errs, k_se=1.0):
    """The tie rule offline: errs[name] = residual array on shared held-out points."""
    names = [n for n in ok if n in errs]
    top = min(names, key=lambda n: np.sqrt(np.mean(errs[n] ** 2)))
    tied = [top]
    for n in names:
        if n == top:
            continue
        d = errs[n] ** 2 - errs[top] ** 2
        if len(d) > 1 and d.mean() <= k_se * d.std(ddof=1) / np.sqrt(len(d)):
            tied.append(n)
    return min(tied, key=SIMPLEST_FIRST.index)


def _rules(rows):
    one = lambda m: m["fold_scores"][0] if m.get("fold_scores") else m["score"]
    rules = [("lowest, score in use", lambda r, ok: min(ok, key=lambda n: ok[n]["score"])),
             ("lowest, one 20% split", lambda r, ok: min(ok, key=lambda n: one(ok[n])))]
    if all("fold_errors" in m for r in rows for m in r["models"].values() if "holdout_true" in m):
        for k in (0.5, 1.0, 2.0):
            rules.append((f"tie rule {k:g} SE, one split", lambda r, ok, k=k: tie_pick(
                ok, {n: np.asarray(ok[n]["fold_errors"][0]) for n in ok}, k)))
        rules.append(("tie rule 1 SE, all folds", lambda r, ok: tie_pick(
            ok, {n: np.concatenate([np.asarray(e) for e in ok[n]["fold_errors"]]) for n in ok}, 1.0)))
    return rules


def _picks(rows, truth_key):
    for label, rule in _rules(rows):
        hits, costs = 0, []
        for r in rows:
            ok = {n: m for n, m in r["models"].items() if "holdout_true" in m}
            pick = rule(r, ok)
            best = min(ok, key=lambda n: ok[n][truth_key])
            hits += pick == best
            if pick != best:
                costs.append(ok[pick][truth_key] / ok[best][truth_key])
        cost = (f"; misses cost {min(costs):.2f}x to {max(costs):.2f}x, "
                f"mean cost over all runs {(sum(costs) + hits) / len(rows):.3f}x") if costs else ""
        print(f"   {label:<30}{hits}/{len(rows)}{cost}")
    print()


def report(files):
    rows = load(files)
    models = list(rows[0]["models"])
    print(f"{len(rows)} runs. Mean error vs the CLEAN truth on held-out points "
          f"(lower is better; noise level: oscillator/missing 0.05, heat 0.02).")
    print(f"wins = runs where that model had the lowest true error.\n")
    pick_report(rows)
    for pname in PROBLEMS:
        for split in ("random", "forecast"):
            sub = [r for r in rows if r["problem"] == pname and r["split"] == split]
            if not sub:
                continue
            key = "dense_true" if all("dense_true" in v for r in sub for v in r["models"].values()
                                      if "holdout_true" in v) else "holdout_true"
            print(f"{pname} / {split}  ({len(sub)} seeds)   columns: held-out points, dense grid"
                  f"{'' if key == 'dense_true' else ' (none)'}; wins by {key}")
            wins = {m: 0 for m in models}
            for r in sub:
                ok = {m: v[key] for m, v in r["models"].items() if "holdout_true" in v}
                if ok:
                    wins[min(ok, key=ok.get)] += 1
            for m in models:
                vals = [r["models"][m].get("holdout_true") for r in sub]
                vals = [v for v in vals if v is not None]
                secs = np.mean([r["models"][m]["seconds"] for r in sub])
                extra = ""
                ps = [r["models"][m].get("params") for r in sub if r["models"][m].get("params")]
                if ps:
                    keys = ps[0].keys()
                    extra = "  " + ", ".join(f"{k}={np.mean([q[k] for q in ps]):.3g}" for k in keys)
                fr = [r["models"][m]["f_rmse"] for r in sub if "f_rmse" in r["models"][m]]
                if fr:
                    extra += f"  f err={np.mean(fr):.2f}"
                dense = [r["models"][m].get("dense_true") for r in sub]
                dense = f"{np.mean(dense):>9.4f}" if vals and None not in dense else f"{'-':>9}"
                if vals:
                    print(f"   {m:<14}{np.mean(vals):>9.4f}{dense}  wins {wins[m]}/{len(sub)}  {secs:6.1f}s{extra}")
                else:
                    print(f"   {m:<14}{'-':>9}   {sub[0]['models'][m]['status'][:70]}")
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="+", metavar="FILE")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--problems", nargs="*", default=list(PROBLEMS), choices=list(PROBLEMS))
    ap.add_argument("--models", nargs="*", default=None, help="default: all")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.report:
        report(a.report)
    else:
        run(a.seeds, a.problems, a.models or "all", Path(a.out))
