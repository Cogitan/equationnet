"""Re-score stage A run 5 after the standard-error fix (third preflight recheck, N29).

Only classical ODE fits whose earliest point was held out moved their start, and only
their standard errors were wrong (too small by sqrt((m - p) / (m - p + ORDER))). This
refits exactly those fits on the same seeds with the fixed code and writes them to
rescore_run5.jsonl, for A3 (main runs) and A4 (coverage). Nothing else is rerun.

    python bench/stageA/rescore_run5.py
"""

import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import equationnet as en  # noqa: E402
from equationnet.fit import _split  # noqa: E402
from problems import COVERAGE_FAMILIES, COVERAGE_SEEDS_RUN5, P, SEEDS_RUN5  # noqa: E402
from run_a import coverage  # noqa: E402

PDE = {"advection_diffusion", "fisher_kpp", "burgers", "heat_edges"}


def gap(name, seed):
    d = P[name]["data"](seed)
    X = np.column_stack([d["t"]])
    train, _ = _split(X, d["u"], ["t"], "random", 0.2, seed, time="t")
    return d["t"][train].min() > d["t"].min()


out = (HERE / "rescore_run5.jsonl").open("w")
for seed in SEEDS_RUN5:
    for name, p in P.items():
        if name in PDE or not p["identifiable"] or not gap(name, seed):
            continue
        res = en.fit(p["equation"], p["data"](seed), split="random", seed=seed,
                     models=["equation_fit"], verbose=False)
        c = res["equation_fit"]
        z = {k: (c.params[k] - v) / c.param_se[k] for k, v in p["true"].items()} if c.ok else None
        out.write(json.dumps({"kind": "main", "problem": name, "seed": seed, "z": z}) + "\n")
        out.flush()
for name in COVERAGE_FAMILIES:
    if name in PDE:
        continue
    for seed in COVERAGE_SEEDS_RUN5:
        if gap(name, seed):
            out.write(json.dumps(dict(coverage(name, seed), kind="coverage")) + "\n")
            out.flush()
print("done")
