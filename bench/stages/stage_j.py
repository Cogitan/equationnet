"""Stage J: your own model (bars in TESTPLAN.md, written before this ran).

    python bench/stages/stage_j.py --out bench/stages/stage_j_results.json

J2  the built-in polynomial's own script, passed in as a model of yours, scores exactly
    what the built-in polynomial scores.
J3  adding models of yours changes no built-in held-out error; when none of yours is
    lowest, the built-in tie marks and the returned model are unchanged.
J4  a model of yours is never returned while a built-in model is tied with it, and the
    table says so when yours had the lowest error.
J5  broken models of yours are reported in words; the built-ins still run.
"""

import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench" / "stageA"))
import equationnet as en  # noqa: E402
from equationnet.fit import _run_script  # noqa: E402
from problems import P  # noqa: E402

SEEDS = range(1100, 1120)   # runs 1 and 2; run 3: --first-seed 1120
NAMES = list(P)


class ScriptModel:
    """A built-in model's own script, used as a model of yours."""
    def __init__(self, script, seed):
        self.script, self.seed = script, seed

    def fit(self, X, y):
        self.ns = _run_script(self.script, "yours")
        self.state = self.ns["train"](X, y, seed=self.seed)
        return self

    def predict(self, X):
        return self.ns["predict"](self.state, X)


class Nearest:
    """Nearest training point, on columns scaled to unit range."""
    def fit(self, X, y):
        self.lo, self.span = X.min(0), np.ptp(X, 0) + 1e-300
        self.X, self.y = (X - self.lo) / self.span, y.copy()
        return self

    def predict(self, X):
        Z = (X - self.lo) / self.span
        return self.y[np.argmin(((Z[:, None, :] - self.X[None]) ** 2).sum(-1), 1)]


def summary(res):
    return {"ranked": [c.name for c in res.ranked], "tied": [c.name for c in res.tied],
            "best": res.best.name if res.ranked else None,
            "scores": {n: c.holdout_rmse for n, c in res.candidates.items() if c.ok},
            "status": {n: c.status for n, c in res.candidates.items()}}


def case(label, equation, data, truth, seed, **kw):
    t0 = time.perf_counter()
    base = en.fit(equation, data, seed=seed, verbose=False, **kw)
    poly_script = base["poly"].script
    yours = [en.custom(ScriptModel(poly_script, seed), name="poly_yours"),
             en.custom(Nearest(), name="nearest")]
    if truth is not None:
        yours.append(en.custom(truth, name="oracle"))
    withy = en.fit(equation, data, seed=seed, verbose=False, models=["all"] + yours, **kw)
    a, b = summary(base), summary(withy)
    built = [n for n in a["scores"]]
    j2 = ("poly" in b["scores"] and "poly_yours" in b["scores"] and
          abs(b["scores"]["poly_yours"] - b["scores"]["poly"]) <= 1e-12 * abs(b["scores"]["poly"]))
    j3_scores = all(a["scores"][n] == b["scores"].get(n) for n in built) and \
        all(a["status"][n] == b["status"][n] for n in a["status"])
    yours_lowest = bool(withy.ranked) and withy.ranked[0].yours
    j3_ties = None
    if not yours_lowest:
        j3_ties = ([n for n in b["tied"] if n in built] == a["tied"]) and b["best"] == a["best"]
    tied = withy.tied
    conflict = any(c.yours for c in tied) and any(not c.yours for c in tied)
    j4 = (not withy.best.yours) if conflict else True
    said = None
    if yours_lowest and conflict:
        said = "Yours has the lowest error, but on a tie a built-in model is" in withy.table()
    return {"case": label, "seed": seed, "kw": {k: v for k, v in kw.items()}, "seconds": round(time.perf_counter() - t0, 1),
            "J2": j2, "J3_scores": j3_scores, "yours_lowest": yours_lowest, "J3_ties": j3_ties,
            "J4_conflict": conflict, "J4": j4, "J4_said": said,
            "poly": b["scores"].get("poly"), "poly_yours": b["scores"].get("poly_yours"),
            "without": a, "with": b}


def broken_case():
    class Raises:
        def fit(self, X, y):
            raise RuntimeError("deliberately broken")

        def predict(self, X):
            return X[:, 0]

    class Short(Nearest):
        def predict(self, X):
            return super().predict(X)[:-1]

    class Nans(Nearest):
        def predict(self, X):
            return np.full(len(X), np.nan)

    try:
        r = en.fit("u_tt + c*u_t + k*u = 0", ROOT / "examples" / "spring.csv", verbose=False,
                   models=["poly", "spline", "equation_fit", Raises(), Short(), Nans()])
        st = {n: c.status for n, c in r.candidates.items()}
        table = r.table()
        ok = (all(st[n] == "ok" for n in ("poly", "spline", "equation_fit")) and
              all(st[n].startswith("failed: ") for n in ("Raises", "Short", "Nans")))
        return {"J5": ok, "status": st, "table": table, "traceback": None}
    except Exception:
        return {"J5": False, "traceback": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-seed", type=int, default=1100)
    args = ap.parse_args()
    rows = []
    spring = ROOT / "examples" / "spring.csv"
    for split in ("random", "forecast"):
        for cv in ((1, 5) if split == "random" else (1, 3)):   # 40 times: 5 forecast origins do not fit
            rows.append(case(f"spring/{split}/cv{cv}", "u_tt + c*u_t + k*u = 0", spring, None, 0,
                             split=split, cv=cv))
            print(rows[-1]["case"], rows[-1]["J2"], rows[-1]["J3_scores"], rows[-1]["J3_ties"],
                  rows[-1]["J4"], rows[-1]["seconds"], flush=True)
    for s in range(args.first_seed, args.first_seed + 20):
        name = NAMES[(s - args.first_seed) % len(NAMES)]
        p = P[name]
        split = "random" if s % 2 == 0 else "forecast"
        rows.append(case(f"{name}/{split}", p["equation"], p["data"](s), p["truth"], s, split=split))
        print(rows[-1]["case"], s, rows[-1]["J2"], rows[-1]["J3_scores"], rows[-1]["J3_ties"],
              rows[-1]["J4"], rows[-1]["seconds"], flush=True)
    j5 = broken_case()
    n = len(rows)
    conflicts = [r for r in rows if r["J4_conflict"]]
    said = [r for r in rows if r["J4_said"] is not None]
    report = {
        "J2": f"{sum(r['J2'] for r in rows)}/{n}",
        "J3_scores": f"{sum(r['J3_scores'] for r in rows)}/{n}",
        "J3_ties": f"{sum(bool(r['J3_ties']) for r in rows if r['J3_ties'] is not None)}/"
                   f"{sum(r['J3_ties'] is not None for r in rows)} (runs where none of yours is lowest)",
        "J4": f"{sum(r['J4'] for r in conflicts)}/{len(conflicts)} runs with a tie between yours and a built-in",
        "J4_said": f"{sum(r['J4_said'] for r in said)}/{len(said)} runs where yours was lowest and tied",
        "J5": j5["J5"],
    }
    print(json.dumps(report, indent=1))
    Path(args.out).write_text(json.dumps({"report": report, "rows": rows, "J5": j5}, indent=1, default=str),
                              encoding="utf-8")


if __name__ == "__main__":
    main()
