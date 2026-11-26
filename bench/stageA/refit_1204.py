"""Stage A run 6: refit advection-diffusion coverage seeds 1203 and 1204 (classical fit only,
as the coverage harness runs it) and record training and held-out errors (TESTPLAN.md run 6).

    python bench/stageA/refit_1204.py
"""
import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import equationnet as en  # noqa: E402
from problems import P  # noqa: E402

p = P["advection_diffusion"]
rows = []
for s in (1203, 1204):
    r = en.fit(p["equation"], p["data"](s), models=["equation_fit"], seed=s, verbose=False)
    c = r["equation_fit"]
    rows.append({"seed": s, "status": c.status, "params": c.params, "se": c.param_se,
                 "train_rmse": c.train_rmse, "holdout_rmse": c.holdout_rmse,
                 "notes": r.constant_notes()})
(HERE / "run6_refit_1204.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(json.dumps(rows, indent=1))
