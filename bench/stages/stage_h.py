"""Stage H of TESTPLAN.md: reproducibility.

Runs fit() with DEFAULT settings on 6 of the stage C equations (decay, logistic,
oscillator, unknown function, heat, Fisher-KPP), in separate processes:
  run a: 1 thread          run b: 1 thread (repeat of a)
  run c: 4 threads         run d: 16 threads
Bars (agreed before running):
  H1  a and b identical: every model's held-out error, constants and predictions
      bit for bit, and the same pick
  H2  a, c, d: same ranking and pick; constants within 1e-6 relative
Also reported (no bar): the largest prediction difference across thread counts.

    python bench/stages/stage_h.py            # writes bench/stages/stage_h_results.json
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
PICK = [0, 1, 2, 3, 6, 8]          # indices into stage_c.CASES


def worker(threads, out):
    import platform
    platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError())
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    import equationnet as en
    from stage_c import CASES
    rows = []
    for i in PICK:
        eq, known, cols = CASES[i]
        target = "T" if eq.startswith("T_") else "u"
        res = en.fit(eq, cols, target=target, known=known, verbose=False)
        indep = [c for c in cols if c != target]
        X = np.column_stack([cols[c] for c in indep])
        row = {"equation": eq, "ranking": [c.name for c in res.ranked], "pick": res.best.name, "models": {}}
        for name, c in res.candidates.items():
            if c.ok:
                row["models"][name] = {"score": c.holdout_rmse, "params": c.params,
                                       "pred": c.predict(X).tolist()}
        rows.append(row)
        print(threads, eq, row["pick"], flush=True)
    Path(out).write_text(json.dumps(rows))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        runs = {"a": 1, "b": 1, "c": 4, "d": 16}
        procs = {}
        for key, n in runs.items():
            env = {**os.environ, "OMP_NUM_THREADS": str(n), "MKL_NUM_THREADS": str(n),
                   "OPENBLAS_NUM_THREADS": str(n)}
            procs[key] = subprocess.Popen([sys.executable, __file__, "--worker", str(n), f"{tmp}/{key}.json"],
                                          env=env)
        for p in procs.values():
            p.wait()
        R = {k: json.loads(Path(f"{tmp}/{k}.json").read_text()) for k in runs}

    import numpy as np
    out = {"H1": [], "H2": []}
    for i, row_a in enumerate(R["a"]):
        eq = row_a["equation"]
        b = R["b"][i]
        same = (row_a["pick"] == b["pick"] and row_a["ranking"] == b["ranking"]
                and all(row_a["models"][m] == b["models"].get(m) for m in row_a["models"]))
        out["H1"].append({"equation": eq, "identical": same})
        for k in ("c", "d"):
            o = R[k][i]
            rel = 0.0
            pred = 0.0
            for m, v in row_a["models"].items():
                w = o["models"].get(m)
                if w is None:
                    rel = float("inf")
                    continue
                for p, x in v["params"].items():
                    rel = max(rel, abs(w["params"][p] - x) / max(abs(x), 1e-12))
                scale = float(np.std(v["pred"])) or 1.0
                pred = max(pred, float(np.max(np.abs(np.subtract(w["pred"], v["pred"])))) / scale)
            out["H2"].append({"equation": eq, "threads": runs[k],
                              "same_ranking": o["ranking"] == row_a["ranking"], "same_pick": o["pick"] == row_a["pick"],
                              "max_rel_param_diff": rel, "max_rel_pred_diff": pred,
                              "ranking_a": row_a["ranking"], "ranking_other": o["ranking"]})
    (HERE / "stage_h_results.json").write_text(json.dumps(out, indent=1))
    print("\nH1 identical:", sum(r["identical"] for r in out["H1"]), "of", len(out["H1"]))
    for r in out["H2"]:
        print(f"H2 {r['equation']:<28} {r['threads']:>2} threads: ranking {'same' if r['same_ranking'] else 'DIFFERENT'}, "
              f"pick {'same' if r['same_pick'] else 'DIFFERENT'}, constants {r['max_rel_param_diff']:.1e}, "
              f"predictions {r['max_rel_pred_diff']:.1e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), sys.argv[3])
    else:
        main()
