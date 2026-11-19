"""Coverage on the diagnosis seeds used after stage A run 2 (TESTPLAN.md, "Fixes after
run 2"): Burgers on 600-619 and 900-939, Duffing on 600-619, all on the run-3 code.
These seeds were used for diagnosis, not for the bars.

    python bench/stageA/diag_coverage.py burgers 600 620 >> bench/stageA/diag_coverage.jsonl
"""

import platform

platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_a import coverage  # noqa: E402

name, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for seed in range(lo, hi):
    print(json.dumps(coverage(name, seed)), flush=True)
