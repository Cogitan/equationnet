"""Stage A run: every problem in problems.py, default settings, both splits.

    python bench/stageA/run_a.py --seeds 11 --out bench/stageA/results_s11.jsonl
    python bench/stageA/run_a.py --coverage --out bench/stageA/coverage.jsonl
    run 2:  --seeds 21 22 23 --out bench/stageA/run2_s21.jsonl ...;  --coverage --run2
    run 3:  --seeds 31 32 33 --out bench/stageA/run3_s31.jsonl ...;  --coverage --run3
    run 4:  --seeds 41 42 43 --out bench/stageA/run4_s41.jsonl ...;  --coverage --run4
    run 5:  --seeds 51 52 53 --out bench/stageA/run5_s51.jsonl ...;  --coverage --run5

Default settings = what a user gets: en.fit(equation, data, split=..., seed=...),
one 20% split, pick="simplest". "Truly best" = the lowest error against the exact
solution on a dense grid (the whole range for random splits, everything after the
last training time for forecasts).
"""

import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import equationnet as en  # noqa: E402
from problems import COVERAGE_FAMILIES, COVERAGE_SEEDS, COVERAGE_SEEDS_RUN2, COVERAGE_SEEDS_RUN3, COVERAGE_SEEDS_RUN4, COVERAGE_SEEDS_RUN5, P  # noqa: E402


def dense_points(X, indep, train, split, time_var):
    lo, hi = X.min(0), X.max(0)
    ti = indep.index(time_var)
    if split == "forecast":
        lo = lo.copy()
        lo[ti] = X[train, ti].max()
    n = 400 if X.shape[1] == 1 else 41
    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi)]
    return np.column_stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")])


def one(name, seed, split):
    p = P[name]
    data = p["data"](seed)
    indep = [c for c in data if c != "u"]
    X = np.column_stack([data[c] for c in indep])
    t0 = time.perf_counter()
    res = en.fit(p["equation"], data, split=split, seed=seed, verbose=False)
    from equationnet.fit import _split
    train, _ = _split(X, data["u"], indep, split, 0.2, seed, time=res.equation.time)
    Q = dense_points(X, indep, train, split, res.equation.time)
    truth = p["truth"](Q)
    row = {"problem": name, "seed": seed, "split": split, "seconds": round(time.perf_counter() - t0, 1),
           "pick": res.best.name if res.ranked else None, "tied": [c.name for c in res.tied],
           "notes": res.constant_notes(), "disagree": res.disagreement_notes(), "table_notes": res.notes, "models": {}}
    for m, c in res.candidates.items():
        r = {"status": c.status, "seconds": round(c.seconds, 2)}
        if c.ok:
            r["score"] = c.holdout_rmse
            r["dense"] = float(np.sqrt(np.mean((c.predict(Q) - truth) ** 2)))
            r["params"] = c.params
            r["se"] = c.param_se
            r["note"] = c.note
        row["models"][m] = r
    return row


def coverage(name, seed):
    p = P[name]
    data = p["data"](seed)
    res = en.fit(p["equation"], data, models=["equation_fit"], seed=seed, verbose=False)
    c = res["equation_fit"]
    out = {"problem": name, "seed": seed, "status": c.status, "z": {}}
    if c.ok:
        for k, v in p["true"].items():
            se = c.param_se.get(k)
            out["z"][k] = (c.params[k] - v) / se if se else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 12, 13])
    ap.add_argument("--problems", nargs="*", default=list(P))
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--run2", action="store_true", help="coverage on the run-2 seeds")
    ap.add_argument("--run3", action="store_true", help="coverage on the run-3 seeds")
    ap.add_argument("--run4", action="store_true", help="coverage on the run-4 seeds")
    ap.add_argument("--run5", action="store_true", help="coverage on the run-5 seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["seed"], r.get("split")))
    with out.open("a") as fh:
        if a.coverage:
            for name in COVERAGE_FAMILIES:
                for seed in (COVERAGE_SEEDS_RUN5 if a.run5 else COVERAGE_SEEDS_RUN4 if a.run4 else COVERAGE_SEEDS_RUN3 if a.run3 else COVERAGE_SEEDS_RUN2 if a.run2 else COVERAGE_SEEDS):
                    if (name, seed, None) in done:
                        continue
                    fh.write(json.dumps(coverage(name, seed)) + "\n")
                    fh.flush()
                    print(name, seed, flush=True)
        else:
            for seed in a.seeds:
                for name in a.problems:
                    for split in ("random", "forecast"):
                        if (name, seed, split) in done:
                            continue
                        try:
                            row = one(name, seed, split)
                        except Exception as exc:          # A1: a stop must be in words; record it
                            row = {"problem": name, "seed": seed, "split": split,
                                   "error": f"{type(exc).__name__}: {exc}"}
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        print(name, seed, split, row.get("pick", row.get("error")), flush=True)
