"""equationnet on real measured data, default settings, both splits.

    python bench/real/run_real.py --out results.jsonl [--datasets yeast coffee ...]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import equationnet as en  # noqa: E402
from datasets import DATASETS  # noqa: E402


def main(names, out, threads):
    torch.set_num_threads(threads)
    for name in names:
        d = DATASETS[name]
        for eq in d["equations"]:
            for split in ("random", "forecast"):
                print(f"\n== {name} / {eq} / {split}", flush=True)
                t0 = time.perf_counter()
                res = en.fit(eq, {"t": d["t"], "u": d["u"]}, split=split, verbose=False)
                print(res.table(), flush=True)
                row = {"dataset": name, "equation": eq, "split": split, "n_train": res.n_train,
                       "n_holdout": res.n_holdout, "pick": res.best.name if res.ranked else None,
                       "lowest": res.ranked[0].name if res.ranked else None,
                       "tied": [c.name for c in res.tied], "seconds": round(time.perf_counter() - t0, 1),
                       "models": {}}
                for m, c in res.candidates.items():
                    row["models"][m] = {"status": c.status, "score": c.holdout_rmse, "params": c.params,
                                        "seconds": round(c.seconds, 2), "note": c.note if c.ok else None,
                                        "errors": c.holdout_errors.tolist() if c.ok else None}
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    main(a.datasets, a.out, a.threads)
