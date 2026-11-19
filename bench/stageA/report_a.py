"""Score stage A against the bars agreed in TESTPLAN.md (written before seeing results).

    python bench/stageA/report_a.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problems import P  # noqa: E402

EQ_MODELS = {"equation_fit", "pinn", "poly_physics"}
GLOB = sys.argv[1] if len(sys.argv) > 1 else "results_s*.jsonl"
rows = [json.loads(line) for f in sorted(HERE.glob(GLOB)) for line in f.read_text().splitlines()]
print(f"{len(rows)} runs\n")

# A1: every model finishes or says why in words
errors = [r for r in rows if "error" in r]
statuses = [(r["problem"], r["seed"], r["split"], m, v["status"]) for r in rows if "models" in r
            for m, v in r["models"].items()]
bad = [s for s in statuses if s[4] != "ok"]
print(f"A1  runs stopped with an error: {len(errors)}; model results not ok: {len(bad)} of {len(statuses)}")
for e in errors:
    print("    RUN ERROR", e["problem"], e["seed"], e["split"], e["error"][:150])
for s in bad:
    print("    ", s[0], s[1], s[2], s[3], "->", s[4][:150])

# A2: default pick within 1.5x of the truly best dense error in >= 90% of runs; no miss over 3x
costs = []
for r in rows:
    if "models" not in r or r["pick"] is None:
        continue
    dense = {m: v["dense"] for m, v in r["models"].items() if "dense" in v and np.isfinite(v["dense"])}
    best = min(dense, key=dense.get)
    costs.append((dense[r["pick"]] / dense[best], r["problem"], r["seed"], r["split"], r["pick"], best))
within = sum(c[0] <= 1.5 for c in costs)
print(f"\nA2  pick within 1.5x of truly best: {within}/{len(costs)} = {within / len(costs):.0%} "
      f"(bar >= 90%); worst {max(c[0] for c in costs):.2f}x (bar <= 3x); exactly best {sum(c[0] == 1 for c in costs)}")
for c in sorted(costs, reverse=True)[:12]:
    if c[0] > 1:
        print(f"     {c[0]:8.2f}x  {c[1]} s{c[2]} {c[3]}: picked {c[4]}, best {c[5]}")

# A3: classical constants within 3 SE of the truth (identifiable problems)
fits, hits, per_const = 0, 0, []
for r in rows:
    p = P[r["problem"]]
    m = r.get("models", {}).get("equation_fit", {})
    if not p["identifiable"] or "params" not in m:
        continue
    ok_all = True
    for k, v in p["true"].items():
        se = m["se"].get(k)
        z = (m["params"][k] - v) / se if se and np.isfinite(se) and se > 0 else np.inf
        per_const.append((abs(z), r["problem"], r["seed"], r["split"], k, m["params"][k], v, se))
        ok_all &= abs(z) <= 3
    fits += 1
    hits += ok_all
print(f"\nA3  classical fits with every constant within 3 SE: {hits}/{fits} = {hits / max(fits, 1):.0%} (bar >= 90%)")
for z in sorted(per_const, reverse=True)[:10]:
    if z[0] > 3:
        print(f"     |z| {z[0]:8.2f}  {z[1]} s{z[2]} {z[3]} {z[4]}: fitted {z[5]:.5g} +- {z[7]:.2g}, true {z[6]:.5g}")

# A5 / A6: "not pinned down" notes
fa = [(r["problem"], r["seed"], r["split"], r["notes"]) for r in rows
      if "notes" in r and P[r["problem"]]["identifiable"] and r["notes"]]
print(f"\nA5  false alarms on identifiable problems: {len(fa)} (bar 0)")
for f in fa:
    print("    ", f[0], f[1], f[2], "|", " / ".join(n[:110] for n in f[3]))
dis = [(r["problem"], r["seed"], r["split"], r["disagree"]) for r in rows if r.get("disagree")]
print(f"    (report only) separate 'models disagree' notes: {len(dis)} runs")
for f in dis:
    print("    ", f[0], f[1], f[2], "|", " / ".join(n[:110] for n in f[3]))
ni = [r for r in rows if "notes" in r and not P[r["problem"]]["identifiable"]]
fired = [r for r in ni if all(any(n.startswith(k + ":") for n in r["notes"]) for k in P[r["problem"]]["true"])]
print(f"A6  non-identifiable runs where every constant is flagged: {len(fired)}/{len(ni)} (bar 100%)")
for r in ni:
    if r not in fired:
        print("     MISSED", r["problem"], r["seed"], r["split"], r["notes"],
              {k: (v, r["models"]["equation_fit"].get("se", {}).get(k)) for k, v in
               r["models"].get("equation_fit", {}).get("params", {}).items()})

# A7: report only
eqbest = 0
n = 0
for r in rows:
    if "models" not in r:
        continue
    dense = {m: v["dense"] for m, v in r["models"].items() if "dense" in v and np.isfinite(v["dense"])}
    n += 1
    eqbest += min(dense, key=dense.get) in EQ_MODELS
print(f"\nA7  an equation model truly best: {eqbest}/{n}")
by = {}
for r in rows:
    if "models" in r:
        dense = {m: v["dense"] for m, v in r["models"].items() if "dense" in v and np.isfinite(v["dense"])}
        by.setdefault((r["problem"], r["split"]), []).append(min(dense, key=dense.get))
for (prob, split), bests in sorted(by.items()):
    print(f"     {prob:<30}{split:<9} truly best: {', '.join(bests)}")

# A4: coverage
cov = HERE / (sys.argv[2] if len(sys.argv) > 2 else "coverage.jsonl")
if cov.exists():
    print("\nA4  coverage of +-1.96 SE (bar 85% to 98% per family):")
    crow = [json.loads(line) for line in cov.read_text().splitlines()]
    for fam in sorted({c["problem"] for c in crow}):
        zs = [z for c in crow if c["problem"] == fam for z in c["z"].values() if z is not None]
        fails = sum(c["status"] != "ok" for c in crow if c["problem"] == fam)
        inside = sum(abs(z) <= 1.96 for z in zs)
        print(f"     {fam:<22} {inside}/{len(zs)} = {inside / max(len(zs), 1):.0%}   rms z {np.sqrt(np.mean(np.square(zs))):.2f}"
              f"   failed fits {fails}")
