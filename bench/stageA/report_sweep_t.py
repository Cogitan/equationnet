"""Summarise stage A run 6 sweep T (report only; TESTPLAN.md "1.1").

    python bench/stageA/report_sweep_t.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
for f in sorted(HERE.glob("run6_sweep_t*.jsonl")):
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    if not rows:
        continue
    n_old = sum(len(r["tied_100"]) for r in rows)
    n_new = sum(len(r["tied"]) for r in rows)
    print(f"{f.name}: {len(rows)} runs, {rows[0]['points']} points, {rows[0]['n_holdout']} held out")
    print(f"  models marked tied: 1.0.0 {n_old}, 1.1 {n_new} (mean {n_old / len(rows):.2f} -> {n_new / len(rows):.2f})")
    print(f"  runs with fewer tied under 1.1: {sum(len(r['tied']) < len(r['tied_100']) for r in rows)}")
    worst_old = max(max(r["score"][m] for m in r["tied_100"]) / min(r["score"].values()) for r in rows)
    worst_new = max(max(r["score"][m] for m in r["tied"]) / min(r["score"].values()) for r in rows)
    print(f"  worst tied model, held-out error / lowest: 1.0.0 {worst_old:.2f}x, 1.1 {worst_new:.2f}x")
    moved = [r for r in rows if r["pick"] != r["pick_100"]]
    print(f"  runs where the returned model differs: {len(moved)}")
    for r in moved:
        d_new, d_old = r["dense"][r["pick"]], r["dense"][r["pick_100"]]
        closer = "1.1" if d_new < d_old else "1.0.0" if d_old < d_new else "same"
        print(f"    seed {r['seed']}: 1.0.0 {r['pick_100']} dense {d_old:.4g}, 1.1 {r['pick']} dense "
              f"{d_new:.4g}; closer to the exact solution: {closer}")
    picks = {}
    for r in rows:
        picks[r["pick"]] = picks.get(r["pick"], 0) + 1
    print(f"  1.1 picks: {picks}")
    fails = [(r["seed"], m, s) for r in rows for m, s in r["status"].items() if s != "ok"]
    print(f"  model results not ok: {len(fails)}")
    for x in fails:
        print("    ", x[0], x[1], x[2][:120])
