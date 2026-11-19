"""Stage G, PDE: equationnet on USCRN soil temperature, default settings, both splits.

    python bench/realG/run_soil.py [--out bench/realG/results_soil.jsonl]

Data from bench/realG/soil.py; expectations in TESTPLAN.md (stage G, PDE), written
before the data file was opened. Also records each model's held-out error per depth.
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
import equationnet as en  # noqa: E402
from equationnet.fit import _split  # noqa: E402

EQ = "u_t = a*u_xx"


def main(out):
    data = np.genfromtxt(HERE / "soil_lincoln_2025.csv", delimiter=",", names=True)
    cols = {"x": data["x"], "t": data["t"], "u": data["u"]}
    X = np.column_stack([cols["x"], cols["t"]])
    for split in ("random", "forecast"):
        print(f"\n== soil / {EQ} / {split}", flush=True)
        t0 = time.perf_counter()
        res = en.fit(EQ, cols, split=split, verbose=False)
        secs = round(time.perf_counter() - t0, 1)
        print(res.table(), flush=True)
        _, test = _split(X, cols["u"], ["x", "t"], split, 0.2, 0)
        depth = cols["x"][test]
        row = {"dataset": "USCRN Lincoln 11 SW 2025 soil", "equation": EQ, "split": split,
               "n_train": res.n_train, "n_holdout": res.n_holdout,
               "pick": res.best.name if res.ranked else None,
               "lowest": res.ranked[0].name if res.ranked else None,
               "tied": [c.name for c in res.tied], "notes": res.constant_notes(),
               "disagree": res.disagreement_notes(), "result_notes": list(res.notes),
               "seconds": secs, "models": {}}
        for m, c in res.candidates.items():
            r = {"status": c.status, "score": c.holdout_rmse if c.ok else None,
                 "train": c.train_rmse if c.ok else None,
                 "params": c.params if c.ok else None, "se": c.param_se if c.ok else None,
                 "note": c.note if c.ok else None, "seconds": round(c.seconds, 1)}
            if c.ok and c.holdout_errors is not None and len(c.holdout_errors) == test.sum():
                e = c.holdout_errors
                r["by_depth"] = {f"{d:g}": float(np.sqrt(np.mean(e[depth == d] ** 2)))
                                 for d in np.unique(depth)}
            row["models"][m] = r
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results_soil.jsonl"))
    main(ap.parse_args().out)
