"""Stage I5, wheel on Python 3.13: stage D installs the wheel on 3.10 and the sdist on
3.13, so this builds the wheel from the repository, installs it into a fresh 3.13
environment outside the repository, and runs the guided session with piped answers on the
packaged example (example data, default columns, damped spring, Standard, Quit).

    python bench/stages/stage_i5_wheel.py WORKDIR     # needs uv; writes stage_i5_wheel.json
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main(work):
    work = Path(work).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(work.parent / "uv-python"))
    env.setdefault("UV_CACHE_DIR", str(work.parent / "uv-cache"))
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                                capture_output=True, text=True).stdout.strip())
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(work / "dist"), str(ROOT)],
                   check=True, capture_output=True, env=env)
    wheel = next((work / "dist").glob("*.whl"))
    venv = work / "v313"
    subprocess.run(["uv", "venv", "--python", "3.13", str(venv)], check=True, capture_output=True, env=env)
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    cli = venv / ("Scripts/equationnet.exe" if os.name == "nt" else "bin/equationnet")
    subprocess.run(["uv", "pip", "install", "--python", str(exe), str(wheel)], check=True,
                   capture_output=True, env=env)
    ver = subprocess.run([str(exe), "-c", "import sys, equationnet; print(sys.version.split()[0]); "
                          "print('site-packages' in equationnet.__file__)"],
                         capture_output=True, text=True, env=env).stdout.split()
    ex = work / "ex"
    ex.mkdir()
    answers = "\n".join(["2", "", "", "4", "n", "1", "1", "7"]) + "\n"
    g = subprocess.run([str(cli)], input=answers, cwd=ex, env=env, capture_output=True, text=True,
                       timeout=1800)
    out = g.stdout + g.stderr
    rec = {"commit": head, "src_has_uncommitted_changes": dirty, "wheel": wheel.name,
           "python": ver[0] if ver else None, "installed_from_site_packages": ver[1:] == ["True"],
           "exit_code": g.returncode, "summary_printed": "What this means" in g.stdout,
           "said_bye": "Bye." in g.stdout, "traceback": "Traceback" in out,
           "pass": g.returncode == 0 and "What this means" in g.stdout and "Bye." in g.stdout
           and "Traceback" not in out, "tail": out[-1500:]}
    text = json.dumps(rec, indent=1)
    for real, shown in ((str(work), "<work>"), (str(ROOT), "<repo>"), (str(Path.home()), "<home>")):
        parts = [re.escape(x) for x in re.split(r"[\\/]+", real) if x]
        text = re.sub(r"[/\x5c]+".join(parts), shown, text, flags=re.IGNORECASE)
    (HERE / "stage_i5_wheel.json").write_text(text, encoding="utf-8")
    print({k: v for k, v in rec.items() if k != "tail"})


if __name__ == "__main__":
    main(sys.argv[1])
