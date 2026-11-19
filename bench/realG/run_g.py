"""Stage G: equationnet on the NIST reference data, default settings, both splits.

    python bench/realG/run_g.py --out bench/realG/results_g.jsonl [--datasets Rat42 ...]

Expectations were written in TESTPLAN.md (stage G) before the first run.
"""

import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import equationnet as en  # noqa: E402
from datasets_g import DATASETS  # noqa: E402


def main(names, out):
    done = set()
    if Path(out).exists():
        done = {(r["dataset"], r["equation"], r["split"]) for r in map(json.loads, Path(out).read_text().splitlines())}
    for name in names:
        d = DATASETS[name]
        for eq in d["equations"]:
            for split in ("random", "forecast"):
                if (name, eq, split) in done:
                    continue
                print(f"\n== {name} / {eq} / {split}", flush=True)
                t0 = time.perf_counter()
                res = en.fit(eq, {d["var"]: d["x"], "u": d["u"]}, split=split, verbose=False)
                print(res.table(), flush=True)
                row = {"dataset": name, "equation": eq, "split": split, "n_train": res.n_train,
                       "n_holdout": res.n_holdout, "pick": res.best.name if res.ranked else None,
                       "lowest": res.ranked[0].name if res.ranked else None,
                       "tied": [c.name for c in res.tied], "notes": res.constant_notes(),
                       "disagree": res.disagreement_notes(), "seconds": round(time.perf_counter() - t0, 1),
                       "models": {}}
                for m, c in res.candidates.items():
                    row["models"][m] = {"status": c.status, "score": c.holdout_rmse if c.ok else None,
                                        "params": c.params if c.ok else None,
                                        "se": c.param_se if c.ok else None, "note": c.note if c.ok else None}
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results_g.jsonl"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    a = ap.parse_args()
    main(a.datasets, a.out)
