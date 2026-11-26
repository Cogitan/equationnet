"""Stage A run 6, sweep T (1.1, report only; TESTPLAN.md "1.1"): the underfitting example
family, built-in models only, under the 1.1 and the 1.0.0 tie rules from the same fit.

u = 1 + 2t + 0.5 sin(25t) + N(0, 0.01), t evenly spaced on [0, 5], 120 points (24 held out)
and 25 points (5 held out), seeds 1300-1319 (noise and split), equation
u_t = 2 + 12.5*b*cos(25*t), default settings otherwise. Dense error: 400 points on [0, 5]
against the exact solution.

    python bench/stageA/sweep_t.py --points 120 --out bench/stageA/run6_sweep_t120.jsonl
    python bench/stageA/sweep_t.py --points 25 --out bench/stageA/run6_sweep_t25.jsonl
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
from run_a import tied_100  # noqa: E402

F = sys.modules["equationnet.fit"]
EQ = "u_t = 2 + 12.5*b*cos(25*t)"
SEEDS = range(1300, 1320)


def exact(t):
    return 1 + 2 * t + 0.5 * np.sin(25 * t)


def one(points, seed):
    t = np.linspace(0, 5, points)
    u = exact(t) + np.random.default_rng(seed).normal(0, 0.01, points)
    t0 = time.perf_counter()
    res = en.fit(EQ, {"t": t, "u": u}, seed=seed, verbose=False)
    q = np.linspace(0, 5, 400)
    dense = {m: float(np.sqrt(np.mean((c.predict(q[:, None]) - exact(q)) ** 2)))
             for m, c in res.candidates.items() if c.ok}
    old = tied_100(res)
    return {"points": points, "seed": seed, "seconds": round(time.perf_counter() - t0, 1),
            "n_holdout": len(res.ranked[0].holdout_errors),
            "tied": [c.name for c in res.tied], "pick": res.best.name,
            "tied_100": [c.name for c in old], "pick_100": min(old, key=F._simplicity).name,
            "score": {c.name: c.holdout_rmse for c in res.ranked},
            "train": {m: c.train_rmse for m, c in res.candidates.items() if c.ok},
            "dense": dense, "status": {m: c.status for m, c in res.candidates.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(x)["seed"] for x in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as fh:
        for seed in SEEDS:
            if seed in done:
                continue
            row = one(a.points, seed)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(a.points, seed, row["pick_100"], "->", row["pick"],
                  len(row["tied_100"]), "->", len(row["tied"]), "tied", flush=True)
