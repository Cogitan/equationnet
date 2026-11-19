"""Stage D of TESTPLAN.md: packaging. Build, install into fresh environments on the
oldest supported Python and the newest available, run every README example as
written (from a folder outside the repository, so only the installed package can be
imported), and run the test suite against the installed package.

    python bench/stages/stage_d.py WORKDIR      # needs uv; writes bench/stages/stage_d_results.json
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHONS = ["3.10", "3.13"]
EQ = "u_tt + c*u_t + k*u = 0"

README_PY = '''import equationnet as en

result = en.fit("u_tt + c*u_t + k*u = 0", "spring.csv")   # columns: t, u
print(result.table())
result.best.save("best_model.py")    # a standalone script: numpy + scipy/torch only
'''
# the README's option list, with its placeholders (equation, data) filled in
README_API = '''import equationnet as en
equation, data = "u_tt + c*u_t + k*u = 0", "spring.csv"
r = en.fit(equation, data,
       target="u",               # target column
       known={"k": 4.0},         # constants you already know
       models=["equation_fit", "pinn"],   # default: all
       split="random",           # or "forecast": hold out the end in time
       holdout=0.2, seed=0,
       cv=1,                     # see "How models are scored"
       predict_at={"t": [12, 14]},   # where you will want predictions (e.g. the future)
       pick="simplest",          # or "lowest"; see "How models are scored"
       config={"pinn": {"steps": 1000}})
print(r.table())
'''
README_CLI = [
    ["fit", EQ, "spring.csv", "--split", "forecast", "--out", "scripts/"],
    ["fit", EQ, "spring.csv", "--cv", "5", "--predict-at", "future_times.csv"],
    ["script", "u_t = D*u_xx", "--model", "pinn"],     # README: > train_pinn.py
    ["models"],
]


def run(cmd, cwd=None, timeout=1800, env=None, stdout_to=None):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env=env)
    if stdout_to is not None:
        Path(stdout_to).write_text(p.stdout)
    return {"cmd": " ".join(map(str, cmd[1:] if len(cmd) > 1 else cmd)), "returncode": p.returncode,
            "tail": (p.stdout + p.stderr)[-1500:]}


def _hide(text, path, shown):
    """Replace a local path in recorded output, however its separators were escaped
    (one, two or four backslashes, or forward slashes)."""
    parts = [re.escape(x) for x in re.split(r"[\\/]+", path) if x]
    return re.sub(r"[/\x5c]+".join(parts), shown, text, flags=re.IGNORECASE)


def main(work):
    work = Path(work).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    # uv's own Python downloads and cache next to the work folder (a sandboxed AppData
    # cannot take uv's rename into its default folder)
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(work.parent / "uv-python"))
    env.setdefault("UV_CACHE_DIR", str(work.parent / "uv-cache"))
    env["COLUMNS"] = "1000"      # keep tools from wrapping paths across lines in the record
    out = {"build": [], "wheel_files": [], "pythons": {}}
    dist = work / "dist"
    out["build"].append(run(["uv", "build", "--out-dir", dist, ROOT], cwd=work, env=env))
    out["build"].append(run(["uvx", "twine", "check", *sorted(dist.glob("*.whl")), *sorted(dist.glob("*.tar.gz"))], cwd=work, env=env))
    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    out["wheel_files"] = zipfile.ZipFile(wheel).namelist()
    for py in PYTHONS:
        r = out["pythons"][py] = {"steps": []}
        venv = work / f"venv{py}"
        exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        cli = venv / ("Scripts/equationnet.exe" if os.name == "nt" else "bin/equationnet")
        step = r["steps"].append
        step(run(["uv", "venv", "--python", py, venv], cwd=work, env=env))
        what = wheel if py == PYTHONS[0] else sdist        # the wheel on one, the sdist on the other
        step(run(["uv", "pip", "install", "--python", exe, what, "pytest"], cwd=work, env=env))
        if any(st["returncode"] for st in r["steps"]):
            r["all_ok"] = False
            continue
        step(run([exe, "-c", "import sys, equationnet; print(sys.version); print(equationnet.__file__)"],
                 cwd=work, env=env))
        ex = work / f"examples{py}"
        ex.mkdir()
        shutil.copy(ROOT / "examples" / "spring.csv", ex)
        (ex / "future_times.csv").write_text("t\n12\n14\n")
        (ex / "readme_example.py").write_text(README_PY)
        (ex / "readme_api.py").write_text(README_API)
        step(run([exe, "readme_example.py"], cwd=ex, env=env))
        step(run([exe, "best_model.py", "spring.csv"], cwd=ex, env=env))
        step(run([exe, "readme_api.py"], cwd=ex, env=env))
        for args in README_CLI:
            target = ex / "train_pinn.py" if args[0] == "script" else None
            step(run([cli, *args], cwd=ex, env=env, stdout_to=target))
        step(run([exe, "-c", "import ast; ast.parse(open('train_pinn.py').read()); print('train_pinn.py parses')"],
                 cwd=ex, env=env))
        # stage I5: the guided session with answers piped in (numbered menus), on the
        # packaged example: example data, default columns, damped spring, quit
        answers = "\n".join(["2", "", "", "4", "n", "1", "1", "7"]) + "\n"   # 7 = Quit
        g = subprocess.run([str(cli)], input=answers, cwd=ex, env=env,
                           capture_output=True, text=True, timeout=1800)
        ok = (g.returncode == 0 and "What this means" in g.stdout and "Bye." in g.stdout
              and "Traceback" not in g.stdout + g.stderr)
        step({"cmd": "equationnet (guided, answers piped)", "returncode": 0 if ok else 1,
              "tail": (g.stdout + g.stderr)[-1500:]})
        tests = work / f"tests{py}"
        shutil.copytree(ROOT / "tests", tests)
        step(run([exe, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib", tests],
                 cwd=work, env=env, timeout=3600))
        r["all_ok"] = all(s["returncode"] == 0 for s in r["steps"])
    out["build_ok"] = all(s["returncode"] == 0 for s in out["build"])
    text = json.dumps(out, indent=1)
    for real, shown in ((str(work), "<work>"), (str(work.parent), "<scratch>"), (str(ROOT), "<repo>"),
                        (tempfile.gettempdir(), "<tmp>"), (str(Path.home()), "<home>")):
        text = _hide(text, real, shown)
    (HERE / "stage_d_results.json").write_text(text)
    print("build ok:", out["build_ok"], "| wheel:", wheel.name, len(out["wheel_files"]), "files")
    for py, r in out["pythons"].items():
        print(py, "all ok:", r["all_ok"])
        for s in r["steps"]:
            print("   ", "ok " if s["returncode"] == 0 else "FAIL", s["cmd"][:110])


if __name__ == "__main__":
    main(sys.argv[1])
