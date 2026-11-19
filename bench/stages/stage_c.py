"""Stage C of TESTPLAN.md: the generated scripts, run on their own.

For every equation below and every model that applies:
  C0  the script fit() returns is byte-identical to the file save_scripts() writes
  C1  it imports nothing but numpy, scipy, torch and the standard library (checked on
      its source), and runs in Python's isolated mode (-I), where equationnet cannot
      be imported, with `python script.py data.csv`
  C2  its predictions match the same script run in the library's environment
      (same seed, same data): deterministic models within 1e-8, network models within
      1e-6 (relative to the data's spread)
Plus the scripts' own CSV reader (report under C1):
  S1  byte-order mark at the start of the file    -> works
  S2  columns in another order, plus an extra one  -> works (extra column ignored, said)
  S3  a missing value                              -> leaves the row out and says so
  S4  text in a number column                      -> stops with a message naming it
  S5  the target column missing                    -> stops with a message naming it
Expected behaviour above was written before the first run (2026-11-16).

    python bench/stages/stage_c.py            # writes bench/stages/stage_c_results.json
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
sys.path.insert(0, str(SRC))
import equationnet as en  # noqa: E402

ALLOWED = {"numpy", "scipy", "torch"} | set(sys.stdlib_module_names)
DETERMINISTIC = {"equation_fit", "spline", "poly"}
FAST = {"pinn": {"steps": 200, "lbfgs": 30}, "poly_physics": {"steps": 200, "lbfgs": 30},
        "nn": {"steps": 200, "lbfgs": 30}}
WMI = "import platform; platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError()); "
rng = np.random.default_rng(0)


def ode(f, y0, t, noise):
    s = solve_ivp(f, (t[0], t[-1]), y0, t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    return s + noise * np.std(s) * rng.standard_normal(len(t))


def grid(fun, nx=11, nt=11, T=0.5, noise=0.01):
    x, t = np.linspace(0, 1, nx), np.linspace(0, T, nt)
    XX, TT = np.meshgrid(x, t)
    U = fun(XX, TT)
    return {"x": XX.ravel(), "t": TT.ravel(), "u": U.ravel() + noise * rng.standard_normal(U.size)}


def burgers_like(X, T):
    # not an exact Burgers solution: a smooth decaying profile, enough to exercise the scripts
    return np.exp(-T) * np.sin(np.pi * X)


t40 = np.linspace(0, 6, 40)
CASES = [
    ("u_t = -k*u", {}, {"t": t40, "u": ode(lambda t, s: [-0.7 * s[0]], [2.0], t40, 0.01)}),
    ("u_t = r*u*(1 - u/K)", {}, {"t": np.arange(19.0),
                                "u": 665 / (1 + (665 / 9.6 - 1) * np.exp(-0.53 * np.arange(19.0)))}),
    ("u_tt + c*u_t + k*u = 0", {}, {"t": t40, "u": ode(lambda t, s: [s[1], -0.3 * s[1] - 4 * s[0]],
                                                       [1.0, 0.0], t40, 0.02)}),
    ("u'' + c*u' + f(u) = 0", {}, {"t": t40, "u": ode(lambda t, s: [s[1], -0.3 * s[1] - 3 * np.tanh(2 * s[0])],
                                                      [1.5, 0.0], t40, 0.02)}),
    ("u_tt = -g*sin(u)", {}, {"t": t40, "u": ode(lambda t, s: [s[1], -9.8 * np.sin(s[0])], [1.0, 0.0],
                                                 t40, 0.01)}),
    ("u_t = -k*(u - A)", {"A": 24.0}, {"t": np.arange(11.0),
                                       "u": 24 + 58.3 * np.exp(-0.043 * np.arange(11.0))}),
    ("u_t = D*u_xx", {}, grid(lambda X, T: np.exp(-0.1 * np.pi ** 2 * T) * np.sin(np.pi * X))),
    ("u_t + u*u_x = nu*u_xx", {}, grid(burgers_like)),
    ("u_t = D*u_xx + r*u*(1 - u)", {}, grid(lambda X, T: 0.5 + 0.3 * np.exp(-T) * np.cos(np.pi * X))),
    ("u_t = -np*exp(-c*t)*u", {}, {"t": t40, "u": ode(lambda t, s: [-0.7 * s[0]], [2.0], t40, 0.01)}),
    ("T_t = -k*(T - 20)", {}, {"t": np.arange(11.0), "T": 20 + 60 * np.exp(-0.05 * np.arange(11.0))}),
]


def write_csv(path, cols, bom=False):
    names = list(cols)
    lines = [",".join(names)] + [",".join(repr(float(cols[n][i])) for n in names)
                                 for i in range(len(cols[names[0]]))]
    path.write_text(("﻿" if bom else "") + "\n".join(lines) + "\n", encoding="utf-8")


def imports_of(src):
    tops = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            tops.add(node.module.split(".")[0])
    return tops


def run_standalone(script, csv):
    """python -I script.py data.csv, with only the WMI workaround added (machine-specific)."""
    boot = WMI + "import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    return subprocess.run([sys.executable, "-I", "-c", boot, str(script), str(csv)],
                          capture_output=True, text=True, timeout=600, cwd=script.parent)


def run_in_library_env(script, csv, out):
    """The same script text, trained the same way, in the dev environment with equationnet importable."""
    code = (WMI + "import sys, numpy as np, equationnet; ns = {'__name__': 'lib'}; "
            "exec(open(sys.argv[1], encoding='utf-8').read(), ns); "
            "X, y = ns['load_csv'](sys.argv[2]); st = ns['train'](X, y); "
            "np.savetxt(sys.argv[3], ns['predict'](st, X))")
    env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-c", code, str(script), str(csv), str(out)],
                          capture_output=True, text=True, timeout=600, env=env)


def main():
    results = {"cases": [], "reader": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, (eq, known, cols) in enumerate(CASES):
            target = "T" if eq.startswith("T_") else "u"
            folder = tmp / f"case{i}"
            folder.mkdir()
            csv = folder / "data.csv"
            write_csv(csv, cols)
            res = en.fit(eq, csv, target=target, known=known, config=FAST, verbose=False)
            saved = {p.stem: p for p in res.save_scripts(folder)}
            for name, c in res.candidates.items():
                row = {"equation": eq, "model": name, "status": c.status}
                if c.script is None:
                    row["result"] = "not applicable"
                    results["cases"].append(row)
                    continue
                script = saved[name]
                row["C0_identical"] = script.read_text(encoding="utf-8") == c.script
                bad = sorted(imports_of(c.script) - ALLOWED)
                row["C1_imports_ok"] = not bad
                row["C1_bad_imports"] = bad
                run = run_standalone(script, csv)
                row["C1_runs"] = run.returncode == 0
                if run.returncode:
                    row["C1_error"] = (run.stderr or run.stdout)[-400:]
                pred_file = folder / f"data_{name}_pred.csv"
                lib_out = folder / f"lib_{name}.txt"
                lib = run_in_library_env(script, csv, lib_out)
                if run.returncode == 0 and lib.returncode == 0:
                    a = np.loadtxt(pred_file, delimiter=",", skiprows=1)[:, -1]
                    b = np.loadtxt(lib_out)
                    scale = float(np.std(cols[target])) or 1.0
                    diff = float(np.max(np.abs(a - b)) / scale)
                    tol = 1e-8 if name in DETERMINISTIC else 1e-6
                    row["C2_max_rel_diff"] = diff
                    row["C2_match"] = diff <= tol
                elif lib.returncode:
                    row["C2_error"] = lib.stderr[-400:]
                results["cases"].append(row)
                print(eq, name, {k: v for k, v in row.items() if k.startswith("C")}, flush=True)

        # the scripts' own CSV reader, on the first case's spline and classical scripts
        eq, known, cols = CASES[0]
        folder = tmp / "reader"
        folder.mkdir()
        write_csv(folder / "data.csv", cols)
        res = en.fit(eq, folder / "data.csv", models=["equation_fit", "spline"], verbose=False)
        scripts = res.save_scripts(folder)
        t, u = cols["t"], cols["u"]
        variants = {
            "S1_bom": ("works", lambda p: write_csv(p, {"t": t, "u": u}, bom=True)),
            "S2_reordered_extra": ("works", lambda p: write_csv(p, {"note": np.zeros(len(t)), "u": u, "t": t})),
            "S3_missing_value": ("says", lambda p: p.write_text(
                "t,u\n" + "".join(f"{a},{'' if i == 5 else b}\n" for i, (a, b) in enumerate(zip(t, u))))),
            "S4_text_value": ("stops", lambda p: p.write_text(
                "t,u\n" + "".join(f"{a},{'abc' if i == 5 else b}\n" for i, (a, b) in enumerate(zip(t, u))))),
            "S5_no_target": ("stops", lambda p: write_csv(p, {"t": t, "y": u})),
        }
        for key, (expect, make) in variants.items():
            p = folder / f"{key}.csv"
            make(p)
            for script in scripts:
                run = run_standalone(script, p)
                out = run.stdout + run.stderr
                traceback = "Traceback" in out
                if expect == "works":
                    ok = run.returncode == 0 and not traceback
                    if key == "S2_reordered_extra":      # expectation: extra column ignored, and said
                        ok = ok and "note" in out
                elif expect == "says":
                    ok = run.returncode == 0 and "left out" in out.lower()
                else:
                    ok = run.returncode != 0 and not traceback and ("abc" in out or "'u'" in out)
                results["reader"].append({"case": key, "script": script.stem, "expected": expect,
                                          "ok": ok, "returncode": run.returncode, "output": out[-300:]})
                print(key, script.stem, "OK" if ok else "MISS", flush=True)
    text = json.dumps(results, indent=1)
    for real, shown in ((tempfile.gettempdir(), "<tmp>"), (str(Path.home()), "<home>")):
        parts = [re.escape(x) for x in re.split(r"[\\/]+", real) if x]   # however it was escaped
        text = re.sub(r"[/\x5c]+".join(parts), shown, text, flags=re.IGNORECASE)
    (HERE / "stage_c_results.json").write_text(text)
    cases = [r for r in results["cases"] if "C1_runs" in r]
    print(f"\nscripts: {len(cases)}  C0 {sum(r['C0_identical'] for r in cases)}  "
          f"C1 imports {sum(r['C1_imports_ok'] for r in cases)}  C1 runs {sum(r['C1_runs'] for r in cases)}  "
          f"C2 match {sum(r.get('C2_match', False) for r in cases)}  "
          f"reader {sum(r['ok'] for r in results['reader'])}/{len(results['reader'])}")


if __name__ == "__main__":
    main()
