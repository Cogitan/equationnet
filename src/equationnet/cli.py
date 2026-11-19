"""Command line:

    equationnet                      (no arguments: a guided session, see guide.py)
    equationnet start                (the same)
    equationnet fit "u_tt + c*u_t + k*u = 0" data.csv [--split forecast] [--out scripts/]
    equationnet script "u_t = D*u_xx" --model pinn > train_pinn.py
    equationnet models
"""

from __future__ import annotations

import argparse
import sys

from .fit import fit, generate
from .models import MODELS, NotApplicable
from .parse import EquationError


def _known(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, eq, v = it.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise ValueError(f"--known takes NAME=NUMBER, e.g. k=4.0; got {it!r}") from None
        if not eq:
            raise ValueError(f"--known takes NAME=NUMBER, e.g. k=4.0; got {it!r}")
    return out


def _cv(text: str):
    if text == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"--cv takes 'auto' or a whole number of folds, not {text!r}") from None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv == ["start"]:
        from .guide import run
        return run()
    p = argparse.ArgumentParser(
        prog="equationnet",
        description="Is your equation better than no equation? Type equationnet with no arguments "
                    "for a guided session, or use the commands below.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start", help="a guided session: questions instead of options (same as no arguments)")

    f = sub.add_parser("fit", help="train every model on your data and rank them")
    f.add_argument("equation")
    f.add_argument("data", help="CSV with one column per variable and one for the target")
    f.add_argument("--target", default="u")
    f.add_argument("--known", nargs="*", metavar="NAME=VALUE", help="constants you already know")
    f.add_argument("--models", nargs="*", choices=list(MODELS), help="default: all")
    f.add_argument("--split", choices=["random", "forecast"], default="random")
    f.add_argument("--holdout", type=float, default=0.2)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--cv", default="1",
                   help="1 = one held-out split (default); k = k-fold cross-validation; 'auto' = only under 50 points")
    f.add_argument("--predict-at", help="CSV of locations you want predictions at (e.g. future times)")
    f.add_argument("--pick", choices=["simplest", "lowest"], default="simplest",
                   help="simplest (default): among models tied within the noise, the one with the "
                        "one the equation constrains most; lowest: the lowest held-out error")
    f.add_argument("--out", help="folder to write every model's script into")

    s = sub.add_parser("script", help="print the training script for one model (no data needed)")
    s.add_argument("equation")
    s.add_argument("--model", choices=list(MODELS), default="pinn")
    s.add_argument("--target", default="u")
    s.add_argument("--vars", nargs="*", help="independent variables, in CSV column order")
    s.add_argument("--known", nargs="*", metavar="NAME=VALUE")

    sub.add_parser("models", help="list the models")

    a = p.parse_args(argv)
    try:
        if a.cmd == "models":
            for m in MODELS.values():
                print(f"{m.name:<14}{m.title}\n{'':<14}" + m.doc.replace("\n", "\n" + " " * 14))
            return 0
        if a.cmd == "script":
            print(generate(a.equation, a.model, target=a.target, indep=a.vars, known=_known(a.known)))
            return 0
        res = fit(a.equation, a.data, target=a.target, known=_known(a.known),
                  models=a.models or "all", split=a.split, holdout=a.holdout, seed=a.seed,
                  cv=_cv(a.cv), predict_at=a.predict_at, pick=a.pick)
        print()
        print(res.table())
        if a.predict_at:
            from pathlib import Path

            import numpy as np

            from .fit import _as_points
            pts = _as_points(a.predict_at, res.equation.indep)
            pred = res.best.predict(pts)
            out = Path(a.predict_at).with_name(Path(a.predict_at).stem + f"_{res.best.name}_pred.csv")
            np.savetxt(out, np.column_stack([pts, pred]), delimiter=",",
                       header=",".join(res.equation.indep + ["prediction"]), comments="")
            print(f"\npredictions of {res.best.title} at {a.predict_at} written to {out}")
            for row, p in list(zip(pts, pred))[:20]:
                print("  " + ", ".join(f"{v:g}" for v in row) + f"  ->  {p:.6g}")
        if a.out:
            paths = res.save_scripts(a.out)
            print(f"\nscripts written: {', '.join(str(x) for x in paths)}")
        return 0
    except EquationError as exc:
        print(f"equation problem: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, NotApplicable) as exc:   # input problems: say it in words
        print(f"problem: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
