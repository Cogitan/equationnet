"""Stage I of TESTPLAN.md: the guided session, on every starter equation.

For each starter: data made by that equation (seeds 900 to 908, never used before),
written to a CSV with real-world column names (units, spaces, a degree sign) and a column
not to use, in an order that is not the equation's. The guided session is driven with
scripted answers, all six models, then the returned model's script is saved.
  I1  the session runs end to end: table printed, no traceback
  I2  the plain-words summary names result.best and says what the ranking and ties say
  I3  the saved script runs on the saved data copy, in Python's isolated mode, and its
      predictions match the same script trained in the library's environment on the
      original file's columns (deterministic models within 1e-8, networks within 1e-6,
      relative to the data's spread)
Bars were written in TESTPLAN.md before this file (2026-11-19).

    python bench/stages/stage_i.py        # writes bench/stages/stage_i_results.json
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from equationnet import guide  # noqa: E402
from test_guide import Scripted  # noqa: E402

WMI = "import platform; platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError()); "
DETERMINISTIC = {"equation_fit", "spline", "poly"}


def ode(rhs, y0, t):
    return solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, rtol=1e-10, atol=1e-12).y[0]


def noisy(u, level, seed):
    return u + level * np.std(u) * np.random.default_rng(seed).standard_normal(u.shape)


def grid(x, t):
    XX, TT = np.meshgrid(x, t)
    return XX.ravel(), TT.ravel()


def cases():
    """(equation, columns as written to the CSV, measured, time, position) per starter."""
    out = []
    t = np.linspace(0, 6, 30)
    out.append(("u_t = -k*u", {"Sample ID": np.arange(30.0), "Concentration (mg/L)":
                noisy(ode(lambda s, y: [-0.5 * y[0]], [3.0], t), 0.02, 900), "Time (min)": t},
                "Concentration (mg/L)", "Time (min)", None))
    t = np.linspace(0, 12, 30)
    out.append(("u_t = k*(L - u)", {"hours": t, "Temperature (°C)":
                noisy(ode(lambda s, y: [0.4 * (80 - y[0])], [20.0], t), 0.02, 901)},
                "Temperature (°C)", "hours", None))
    t = np.linspace(0, 12, 40)
    out.append(("u_t = r*u*(1 - u/K)", {"Population": noisy(ode(lambda s, y: [0.8 * y[0] * (1 - y[0] / 50)],
                [2.0], t), 0.03, 902), "Day": t}, "Population", "Day", None))
    t = np.linspace(0, 10, 50)
    out.append(("u_tt + c*u_t + k*u = 0", {"t [s]": t, "displacement [m]":
                noisy(ode(lambda s, y: [y[1], -0.2 * y[1] - 3 * y[0]], [1.0, 0.0], t), 0.03, 903)},
                "displacement [m]", "t [s]", None))
    t = np.linspace(0, 6, 60)
    out.append(("u_tt = -w*sin(u)", {"time": t, "angle (rad)":
                noisy(ode(lambda s, y: [y[1], -9.81 * np.sin(y[0])], [1.2, 0.0], t), 0.02, 904)},
                "angle (rad)", "time", None))
    x, tt = grid(np.linspace(0, 1, 21), np.linspace(0, 2, 11))
    out.append(("u_t = D*u_xx", {"time (days)": tt, "depth (m)": x, "temperature":
                noisy(np.exp(-0.05 * np.pi ** 2 * tt) * np.sin(np.pi * x), 0.01, 905)},
                "temperature", "time (days)", "depth (m)"))
    x, tt = grid(np.linspace(0, 2, 21), np.linspace(0, 1, 11))
    w = 0.01 + 4 * 0.03 * tt
    out.append(("u_t + v*u_x = D*u_xx", {"distance km": x, "dye level": noisy(
                np.sqrt(0.01 / w) * np.exp(-(x - 0.5 - 0.6 * tt) ** 2 / w), 0.01, 906), "t (h)": tt},
                "dye level", "t (h)", "distance km"))
    D, r = 0.02, 1.5
    out.append(("u_t = D*u_xx + r*u*(1 - u)", {"position": x, "time": tt, "density": noisy(
                1.0 / (1 + np.exp(np.sqrt(r / (6 * D)) * (x - 1.0) - 5 * r / 6 * tt)) ** 2, 0.01, 907)},
                "density", "time", "position"))
    x, tt = grid(np.linspace(0, 1, 21), np.linspace(0, 1, 11))
    nu, a = 0.08, 2.0
    e = np.exp(-np.pi ** 2 * nu * tt)
    out.append(("u_t + u*u_x = nu*u_xx", {"x (m)": x, "t (s)": tt, "velocity (m/s)": noisy(
                2 * nu * np.pi * e * np.sin(np.pi * x) / (a + e * np.cos(np.pi * x)), 0.01, 908)},
                "velocity (m/s)", "t (s)", "x (m)"))
    return out


def write_csv(path, cols):
    names = list(cols)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(f'"{n}"' if "," in n else n for n in names) + "\n")
        for row in zip(*cols.values()):
            fh.write(",".join(repr(float(v)) for v in row) + "\n")


def expected_phrase(res):
    ranked, tied = res.ranked, {c.name for c in res.tied}
    eqm = [c for c in ranked if c.uses_equation]
    dom = [c for c in ranked if not c.uses_equation]
    if not eqm:
        return "None of the models that use your equation finished"
    if not dom:
        return "Only models that use the equation finished"
    e, n = eqm[0], dom[0]
    if e.name in tied and n.name in tied:
        return "This data cannot tell whether your equation helps"
    return "Your equation helped" if e.holdout_rmse < n.holdout_rmse else "Your equation did not help"


def run_standalone(script, csv):
    boot = WMI + "import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    return subprocess.run([sys.executable, "-I", "-c", boot, str(script), str(csv)],
                          capture_output=True, text=True, timeout=1200, cwd=script.parent)


def run_in_library_env(script, original, letters, out):
    """The same script, trained in the dev environment on the ORIGINAL file's columns
    (read by equationnet, mapped to letters), not on the copy."""
    code = (WMI + "import sys, json, numpy as np\n"
            "from equationnet import guide\n"
            "cols, err = guide.load_table(sys.argv[2]); assert err is None, err\n"
            "letters = json.loads(sys.argv[3])\n"
            "ns = {'__name__': 'lib'}; exec(open(sys.argv[1], encoding='utf-8').read(), ns)\n"
            "need = ns['INDEP'] + [ns['TARGET']]\n"
            "A = np.column_stack([cols[letters[k]] for k in need]); A = A[np.all(np.isfinite(A), 1)]\n"
            "X, y = A[:, :-1], A[:, -1]; st = ns['train'](X, y)\n"
            "np.savetxt(sys.argv[4], ns['predict'](st, X))\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-c", code, str(script), str(original), json.dumps(letters),
                           str(out)], capture_output=True, text=True, timeout=1200, env=env)


def main():
    results = []
    real_fit = guide.fit
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, (eq, cols, measured, timecol, position) in enumerate(cases()):
            folder = tmp / f"case{i}"
            folder.mkdir()
            csv = folder / f"field data {i}.csv"
            write_csv(csv, cols)
            got = {}

            def capture(*a, **k):
                got["res"] = real_fit(*a, **k)
                return got["res"]

            guide.fit = capture
            answers = ["file", str(csv), measured, timecol]
            if len(cols) > 2:
                answers.append(position)
            split = "forecast" if i % 2 else "random"
            answers += [eq, False, split, 1, "best", str(folder / "out"), "quit"]
            a = Scripted(answers)
            t0 = time.time()
            row = {"equation": eq, "columns": list(cols), "split": split}
            try:
                code = guide.run(a)
                err = None
            except Exception as exc:          # a traceback is a miss (I1)
                code, err = None, f"{type(exc).__name__}: {exc}"
            guide.fit = real_fit
            out = a.text_out
            row["seconds"] = round(time.time() - t0, 1)
            res = got.get("res")
            row["I1_runs"] = code == 0 and err is None and res is not None and "rank model" in out \
                and "Traceback" not in out and not a.rejected
            if err:
                row["I1_error"] = err
            if a.rejected:
                row["I1_rejected"] = a.rejected
            if res is not None and res.ranked:
                phrase = expected_phrase(res)
                row["returned"] = res.best.name
                row["ranking"] = [(c.name, round(c.holdout_rmse, 6)) for c in res.ranked]
                row["tied"] = [c.name for c in res.tied]
                row["expected_phrase"] = phrase
                row["I2_summary_ok"] = phrase in out and f"Returned: {res.best.title}" in out
                summary = out[out.find("What this means"):]
                row["summary"] = summary[:1200]
                script = folder / "out" / f"{res.best.name}.py"
                copies = list((folder / "out").glob("*_equationnet.csv"))
                if script.exists() and copies:
                    run = run_standalone(script, copies[0])
                    row["I3_runs"] = run.returncode == 0
                    if run.returncode:
                        row["I3_error"] = (run.stderr or run.stdout)[-600:]
                    letters = {"x": position, "t": timecol, "u": measured}
                    lib_out = folder / "lib.txt"
                    lib = run_in_library_env(script, csv, {k: v for k, v in letters.items() if v}, lib_out)
                    pred = copies[0].with_name(f"{copies[0].stem}_{script.stem}_pred.csv")
                    if run.returncode == 0 and lib.returncode == 0:
                        p = np.loadtxt(pred, delimiter=",", skiprows=1)[:, -1]
                        q = np.loadtxt(lib_out)
                        scale = float(np.std(cols[measured])) or 1.0
                        diff = float(np.max(np.abs(p - q)) / scale)
                        tol = 1e-8 if res.best.name in DETERMINISTIC else 1e-6
                        row["I3_max_rel_diff"] = diff
                        row["I3_match"] = diff <= tol
                    elif lib.returncode:
                        row["I3_lib_error"] = lib.stderr[-600:]
                else:
                    row["I3_runs"] = False
                    row["I3_error"] = "script or data copy not saved"
            results.append(row)
            print(eq, {k: v for k, v in row.items() if k[:2] in ("I1", "I2", "I3")},
                  row.get("returned"), row.get("expected_phrase"), f"{row['seconds']}s", flush=True)
    text = json.dumps(results, indent=1, ensure_ascii=False)
    for real, shown in ((str(ROOT), "<repo>"), (tempfile.gettempdir(), "<tmp>"), (str(Path.home()), "<home>")):
        parts = [re.escape(x) for x in re.split(r"[\\/]+", real) if x]
        text = re.sub(r"[/\x5c]+".join(parts), shown, text, flags=re.IGNORECASE)
    (HERE / "stage_i_results.json").write_text(text, encoding="utf-8")
    n = len(results)
    print(f"I1 {sum(r['I1_runs'] for r in results)}/{n}  "
          f"I2 {sum(r.get('I2_summary_ok', False) for r in results)}/{n}  "
          f"I3 runs {sum(r.get('I3_runs', False) for r in results)}/{n}, "
          f"match {sum(r.get('I3_match', False) for r in results)}/{n}")


if __name__ == "__main__":
    main()
