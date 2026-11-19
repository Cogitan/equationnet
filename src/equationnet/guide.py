"""Guided session: type `equationnet` with no arguments.

It asks for the data, what each column is, and the equation (a starter or your own),
runs the same `fit()` as the Python API, and explains the result in plain words. No
flags, no file formats beyond CSV, no knowledge of the models needed.

In a terminal it uses arrow-key menus (questionary). Without one (answers piped in,
or questionary missing) it falls back to numbered menus read line by line.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fit import MIN_POINTS, _with_se, fit
from .models import VERSION, NotApplicable
from .parse import EquationError, parse_equation
from .table import read_table

# Starter equations, written for the letters the guided session gives the columns:
# u = what was measured, t = time, x = position. Every one is a stage A problem
# (TESTPLAN.md) or the README's own example, except "Relaxing toward a level", which is
# the NIST BoxBOD model from stage G.
STARTERS_1 = [
    ("Steady decay or growth", "u_t = -k*u",
     "u shrinks toward zero in proportion to itself (or grows, with k < 0): radioactive decay, "
     "a drug clearing the blood, compound growth"),
    ("Relaxing toward a level", "u_t = k*(L - u)",
     "u moves toward a level L, fast when far from it: something cooling to room temperature "
     "(Newton's law of cooling), warming up, charging, oxygen demand"),
    ("Logistic growth", "u_t = r*u*(1 - u/K)",
     "growth that levels off at a ceiling K: populations, adoption, cell cultures"),
    ("Damped spring", "u_tt + c*u_t + k*u = 0",
     "an oscillation that dies away: springs, circuits, a swaying structure"),
    ("Pendulum", "u_tt = -w*sin(u)",
     "a swinging pendulum at any amplitude (u is the angle in radians)"),
]
STARTERS_2 = [
    ("Heat or diffusion", "u_t = D*u_xx",
     "spreads out evenly: heat in a rod or in soil, dye in still water"),
    ("Diffusion with drift", "u_t + v*u_x = D*u_xx",
     "spreads out while being carried along at speed v: a pollutant in a river"),
    ("Spreading with growth", "u_t = D*u_xx + r*u*(1 - u)",
     "spreads and grows toward 1 (u scaled so its ceiling is 1): an invasion front"),
    ("Burgers' equation", "u_t + u*u_x = nu*u_xx",
     "a wave that steepens, smoothed by viscosity nu: traffic, shock fronts"),
]

EQUATION_HELP = """\
How to write an equation
  u_t      how fast u changes over time (du/dt)
  u_tt     the second derivative over time (acceleration)
  u_x      how u changes with position; u_xx the second derivative
  + - * /  as usual; write k*u, not ku.  ^ or ** for powers
  sin cos exp log sqrt tanh abs pi   (and the other usual math functions)
  f(u)     an unknown function of one argument, learned from the data
  any other name (k, D, rate, ...) is a constant to fit

  Examples:  u_t = -k*u      u_tt + c*u_t + k*u = 0      u_t = D*u_xx
"""

EXAMPLE_NOTE = ("Example data: 40 noisy measurements of a damped spring (columns t and u), "
                "made with c = 0.3 and k = 4 in u_tt + c*u_t + k*u = 0, noise 0.05.")


class Quit(Exception):
    """The user left (Ctrl-C, Esc, or end of input)."""


# ---------------------------------------------------------------- askers

class PlainAsker:
    """Numbered menus and typed answers, one line each. Works anywhere, including with
    answers piped in."""

    def __init__(self, read=input, write=print):
        self._read, self._write = read, write

    def out(self, text: str = "", style: str | None = None) -> None:
        self._write(text)

    def _line(self, prompt: str) -> str:
        try:
            return self._read(prompt)
        except (EOFError, KeyboardInterrupt):
            raise Quit from None

    def select(self, msg, choices, default=None):
        self.out(msg)
        for i, (label, _) in enumerate(choices, 1):
            self.out(f"  {i}. {label}")
        dflt = next((i for i, (_, v) in enumerate(choices, 1) if v == default), None)
        while True:
            a = self._line(f"number{f' [{dflt}]' if dflt else ''}: ").strip()
            if not a and dflt:
                return choices[dflt - 1][1]
            if a.isdigit() and 1 <= int(a) <= len(choices):
                return choices[int(a) - 1][1]
            self.out(f"  type a number from 1 to {len(choices)}")

    def text(self, msg, default="", validate=None):
        while True:
            ask = msg[:-1] if msg.endswith(":") else msg
            a = self._line(f"{ask}{f' [{default}]' if default else ''}: ").strip() or default
            err = validate(a) if validate else None
            if err is None:
                return a
            self.out(f"  {err}")

    def confirm(self, msg, default=True):
        while True:
            a = self._line(f"{msg} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
            if not a:
                return default
            if a in ("y", "yes"):
                return True
            if a in ("n", "no"):
                return False
            self.out("  type y or n")

    def checkbox(self, msg, choices):
        self.out(msg)
        for i, (label, _) in enumerate(choices, 1):
            self.out(f"  {i}. {label}")
        while True:
            a = self._line("numbers separated by commas (empty for none): ").strip()
            if not a:
                return []
            parts = [p.strip() for p in a.replace(" ", ",").split(",") if p.strip()]
            if all(p.isdigit() and 1 <= int(p) <= len(choices) for p in parts):
                return [choices[int(p) - 1][1] for p in dict.fromkeys(parts)]
            self.out(f"  type numbers from 1 to {len(choices)}, separated by commas")

    def path(self, msg, validate=None):
        return self.text(msg, validate=validate)


class MenuAsker:
    """Arrow-key menus in a real terminal (questionary)."""

    def __init__(self):
        import questionary
        self.q = questionary
        self.style = questionary.Style([("qmark", "fg:#5fd3a0 bold"), ("question", "bold"),
                                        ("pointer", "fg:#5fd3a0 bold"), ("highlighted", "fg:#5fd3a0 bold"),
                                        ("selected", "fg:#5fd3a0"), ("answer", "fg:#5fd3a0"),
                                        ("instruction", "fg:#7d838c")])

    def out(self, text: str = "", style: str | None = None) -> None:
        if style:
            self.q.print(text, style={"title": "bold", "dim": "fg:#7d838c", "good": "fg:#5fd3a0 bold",
                                      "warn": "fg:#e6be6e"}.get(style, style))
        else:
            print(text)

    def _ask(self, question):
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
        kb = KeyBindings()

        @kb.add("escape", eager=True)          # Esc leaves, like Ctrl-C
        def _(event):
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

        app = question.application
        app.key_bindings = merge_key_bindings([app.key_bindings, kb]) if app.key_bindings else kb
        a = question.ask()
        if a is None:
            raise Quit
        return a

    @staticmethod
    def _v(validate):
        if validate is None:
            return None
        return lambda a: True if validate(a.strip()) is None else validate(a.strip())

    def select(self, msg, choices, default=None):
        cs = [self.q.Choice(label, value=i) for i, (label, _) in enumerate(choices)]
        d = next((c for c, (_, v) in zip(cs, choices) if v == default), None)
        return choices[self._ask(self.q.select(msg, cs, default=d, style=self.style,
                                               instruction="(arrow keys, Enter)"))][1]

    def text(self, msg, default="", validate=None):
        return self._ask(self.q.text(msg, default=default, validate=self._v(validate),
                                     style=self.style)).strip()

    def confirm(self, msg, default=True):
        return self._ask(self.q.confirm(msg, default=default, style=self.style))

    def checkbox(self, msg, choices):
        cs = [self.q.Choice(label, value=i) for i, (label, _) in enumerate(choices)]
        got = self._ask(self.q.checkbox(msg, cs, style=self.style,
                                        instruction="(Space to tick, Enter to go on)"))
        return [choices[i][1] for i in got]

    def path(self, msg, validate=None):
        return self._ask(self.q.path(msg, validate=self._v(validate), style=self.style)).strip()


def default_asker():
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return MenuAsker()
        except ImportError:
            pass
    return PlainAsker()


# ---------------------------------------------------------------- the session

@dataclass
class Data:
    source: str                         # file path, or "example"
    cols: dict                          # column name -> values (all columns of the file)
    measured: str = ""
    time: str = ""
    position: str | None = None
    unused: list = field(default_factory=list)

    @property
    def letters(self) -> dict:
        """letter -> column name, in the order fit() takes them (variables, then u)."""
        m = {}
        if self.position:
            m["x"] = self.position
        m["t"] = self.time
        m["u"] = self.measured
        return m

    @property
    def indep(self) -> list[str]:
        return [k for k in self.letters if k != "u"]

    def columns(self) -> dict:
        return {k: self.cols[v] for k, v in self.letters.items()}

    @property
    def renamed(self) -> bool:
        return any(k != v for k, v in self.letters.items())

    def name(self, letter: str) -> str:
        return self.letters.get(letter, letter)


@dataclass
class Choice:
    equation: str = ""
    known: dict = field(default_factory=dict)
    split: str = "random"
    cv: int = 1


def clean_path(text: str) -> str:
    """A path as typed or dragged into a terminal: surrounding quotes and spaces go;
    a leading & from PowerShell's drag-and-drop too."""
    t = text.strip()
    if t.startswith("& "):
        t = t[2:].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        t = t[1:-1]
    return t


def load_table(text: str):
    """(columns, None) or (None, a message in words)."""
    p = Path(clean_path(text)).expanduser()
    if not clean_path(text):
        return None, "type the path of a CSV file, or drag the file into this window"
    if p.is_dir():
        return None, f"{p} is a folder; pick a .csv file inside it"
    if p.suffix.lower() in (".xlsx", ".xls", ".ods", ".numbers"):
        return None, ("equationnet reads CSV files. In Excel or Sheets: File > Save As (or Download) "
                      "> CSV, then pick that file")
    try:
        cols = read_table(p)
    except UnicodeDecodeError:
        return None, f"{p.name} is not a text file; equationnet reads CSV (comma-separated) files"
    except (OSError, ValueError) as exc:
        return None, str(exc)
    if len(cols) < 2:
        why = "".join(f"; '{k}' is not used ({v})" for k, v in cols.skipped.items())
        return None, (f"{p.name} has {['no', 'one'][len(cols)]} column{'' if len(cols) == 1 else 's'} of numbers "
                      f"({', '.join(cols) or 'none'}){why}. It needs at least two: what you measured, and "
                      "the time (or position) of each measurement")
    full = [n for n, v in cols.items() if np.isfinite(v).sum() >= MIN_POINTS]
    if len(full) < 2:
        return None, f"{p.name} needs at least {MIN_POINTS} values in at least two columns"
    return cols, None


def _example_cols():
    from importlib import resources
    text = resources.files("equationnet").joinpath("data/spring.csv").read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))
    return {h: np.array([float(r[i]) for r in rows[1:]]) for i, h in enumerate(rows[0])}


SHORT = {"equation_fit": "the classical fit", "poly_physics": "the polynomial held to the equation",
         "pinn": "the PINN", "poly": "the polynomial", "spline": "the smoothing spline",
         "nn": "the neural network"}


def _short(c) -> str:
    return SHORT.get(c.name, c.title)


def _fmt(v) -> str:
    return f"{v:.4g}"


def _range(v) -> str:
    f = v[np.isfinite(v)]
    return f"{_fmt(f.min())} to {_fmt(f.max())}" if len(f) else "no values"


def ask_data(a) -> Data:
    how = a.select("Where is your data?", [
        ("A CSV file on this computer", "file"),
        ("Try the example first (a damped spring, 40 points)", "example")])
    if how == "example":
        a.out(EXAMPLE_NOTE, "dim")
        return Data("example", _example_cols())
    a.out("Tip: you can drag the file into this window instead of typing its path.", "dim")
    holder = {}

    def check(text):
        cols, err = load_table(text)
        if err is None:
            holder["cols"] = cols
        return err

    path = a.path("CSV file:", validate=check)
    if "cols" not in holder:            # an asker that skipped validation
        holder["cols"], err = load_table(path)
        if err:
            raise ValueError(err)
    return Data(str(Path(clean_path(path)).expanduser()), holder["cols"])


def show_data(a, d: Data) -> None:
    n = len(next(iter(d.cols.values())))
    a.out("")
    a.out(f"{n} rows. Columns:", "title")
    w = max(len(c) for c in [*d.cols, *getattr(d.cols, "skipped", {})])
    for c, v in d.cols.items():
        miss = int((~np.isfinite(v)).sum())
        a.out(f"  {c:<{w}}  {_range(v)}" + (f"   ({miss} empty)" if miss else ""))
    for c, why in getattr(d.cols, "skipped", {}).items():
        a.out(f"  {c:<{w}}  not used: {why}", "dim")
    for note in getattr(d.cols, "notes", []):
        a.out(f"  note: {note}", "warn")
    a.out("")


TIME_WORDS = ("time", "sec", "second", "min", "hour", "hr", "day", "week", "month", "year",
              "date", "elapsed", "age")


def _time_guess(cols: dict) -> str:
    """The column most likely to be time: a time-like name, else one that only increases."""
    names = list(cols)
    for c in names:
        words = "".join(ch if ch.isalpha() else " " for ch in c.lower()).split()
        if c.lower() == "t" or (words and words[0] == "t") or \
                any(w.startswith(tw) for w in words for tw in TIME_WORDS):
            return c
    for c in names:
        f = cols[c][np.isfinite(cols[c])]
        if len(f) > 2 and np.all(np.diff(f) > 0):
            return c
    return names[0]


def _measured_guess(cols: dict, time: str) -> str:
    """The column that changes most relative to its size, among the rest (a column that is
    nearly constant, like a room temperature beside a cooling cup, is rarely the one)."""
    rest = [c for c in cols if c != time] or list(cols)
    for c in rest:
        if c.lower() in ("u", "y"):
            return c

    def spread(c):
        f = cols[c][np.isfinite(cols[c])]
        return float(np.std(f) / (abs(np.mean(f)) + np.std(f) + 1e-300)) if len(f) else 0.0
    return max(rest, key=spread)


def ask_roles(a, d: Data) -> None:
    names = list(d.cols)
    lab = lambda c: f"{c}   ({_range(d.cols[c])})"
    tguess = _time_guess(d.cols)
    d.measured = a.select("Which column did you measure? (the quantity your equation is about)",
                          [(lab(c), c) for c in names], default=_measured_guess(d.cols, tguess))
    rest = [c for c in names if c != d.measured]
    d.time = a.select("Which column is time? (or, if nothing changes in time, the one thing "
                      "the measurement changes along)", [(lab(c), c) for c in rest],
                      default=tguess if tguess in rest else _time_guess({c: d.cols[c] for c in rest}))
    rest = [c for c in rest if c != d.time]
    d.position = None
    if rest:
        d.position = a.select(
            f"Was {d.measured} also measured at different positions (depth, distance, ...), "
            f"so that it changes with both {d.time} and position? Most data is not like this.",
            [(f"No: {d.measured} changes only with {d.time}", None)] +
            [(f"Yes: {c} is the position", c) for c in rest], default=None)
    unused = [c for c in rest if c != d.position]
    d.unused = unused
    a.out("")
    a.out("In equations your columns are called:", "title")
    for k, v in d.letters.items():
        a.out(f"  {k} = {v}")
    if unused:
        a.out(f"  not used: {', '.join(unused)}", "dim")
    a.out("")


def ask_equation(a, d: Data) -> str:
    starters = STARTERS_2 if d.position else STARTERS_1
    w = max(len(s[0]) for s in starters)
    while True:
        pick = a.select("Which equation should be tested against your data?",
                        [(f"{t:<{w}}   {e}", e) for t, e, _ in starters] +
                        [("Type my own equation", "own"), ("How do I write an equation?", "help")])
        if pick == "help":
            a.out(EQUATION_HELP)
            continue
        if pick != "own":
            title, _, what = next(s for s in starters if s[1] == pick)
            a.out(f"{title}: {what}.", "dim")
            return pick
        names = ", ".join(f"{k} for {v}" for k, v in d.letters.items())
        a.out(f"Use {names}. Examples: {starters[0][1]}   {starters[-1][1]}", "dim")

        def check(text):
            try:
                parse_equation(text, target="u", indep=d.indep)
            except EquationError as exc:
                msg = str(exc)
                others = [n for n, _ in re.findall(r"\b([A-Za-z]\w*)_([A-Za-z]+)\b", text) if n != "u"]
                if others:
                    msg += (f". Here the column you measured ({d.measured}) is called u, so write "
                            f"u_{d.indep[-1]} rather than {others[0]}_{d.indep[-1]}")
                return msg
            return None

        text = a.text("Equation:", validate=check)
        eq = parse_equation(text, target="u", indep=d.indep)
        a.out(f"Read as: {eq.describe()}", "dim")
        if eq.ufuncs:
            fs = ", ".join(f"{u.name}({u.arg})" for u in eq.ufuncs)
            first = eq.ufuncs[0].name
            if not a.confirm(f"{fs}: a name followed by brackets is an unknown function, learned from "
                             f"the data. Did you mean that? (To multiply, write {first}*(...).)",
                             default=False):
                continue
        return text


def ask_known(a, d: Data, equation: str) -> dict:
    params = parse_equation(equation, target="u", indep=d.indep).params
    if not params:
        return {}
    a.out(f"Constants to fit: {', '.join(params)}", "dim")
    for c in d.unused:
        f = d.cols[c][np.isfinite(d.cols[c])]
        if len(f):
            a.out(f"If your column '{c}' measures one of these (say, the level something settles "
                  f"to), you can give its average, {np.mean(f):.4g}, as that constant's value.", "dim")
    if not a.confirm("Do you already know the value of any of them?", default=False):
        return {}
    chosen = a.checkbox("Which ones do you know?", [(p, p) for p in params])

    def number(text):
        try:
            v = float(text)
        except ValueError:
            return f"type a number, e.g. 4.0 or 1.5e-3 (not {text!r})"
        return None if np.isfinite(v) else "type a finite number"

    return {p: float(a.text(f"{p} =", validate=number)) for p in chosen}


def ask_test(a, d: Data) -> tuple[str, int]:
    split = a.select("What should the models be tested on?", [
        ("Points scattered through the data (how well each fills in gaps)", "random"),
        (f"The end of the data in {d.time} (how well each predicts what comes next)", "forecast")])
    careful = (("Careful: 5 rounds, every point held back once (trains each model 6 times "
                "instead of once; steadier on small data)", 5) if split == "random" else
               ("Careful: 3 forecast windows, each trained only on what came before it (trains "
                "each model 4 times instead of once; steadier on small data)", 3))
    held = "the last 20% of the times" if split == "forecast" else "20% of the points"
    cv = a.select("How thorough?", [
        (f"Standard: hold back {held} once and score every model on them", 1), careful])
    return split, cv


LEGEND = ("How to read it: held-out error is how far off each model was, typically, on points it "
          "never saw (same units as u; lower is better). '=' marks models the data cannot tell apart. "
          "+- is one standard error (roughly a 68% range), if the equation is exactly right.")

GLOSSARY = """\
Words used here
  held-out points   data hidden from every model while it trains, then used to score it
  held-out error    how far off a model was on those points, typically (root mean square)
  data-only model   fits a curve to the data and ignores the equation
  classical fit     solves your equation with a standard solver and fits its constants
  PINN              a neural network trained to match the data and obey the equation
  polynomial held to the equation   a polynomial curve trained to match the data and obey the equation
  smoothing spline  a smooth curve through the data, a data-only model
  training points   the points the models learn from (everything not held out)
  forecast window   a stretch of time held out at the end, after the training points
  within the noise  the difference is too small to mean anything with this many points
  constrains most   when models tie, the one with the fewest free choices is returned: the
                    classical fit, then the polynomial held to the equation, then the PINN,
                    then the data-only models
  standard error    the +- after a constant: roughly a 68% range, if the equation is right
  scattered / end   test on points spread through the data, or on the last part in time
  Careful           repeat the test several times on different held-out points and average
"""


def guided_table(res) -> str:
    """The table, with its Python-only advice put in this session's terms."""
    text = re.sub(r"pass known=\{'(\w+)': \.\.\.\}",
                  r"give \1 when asked which constants you already know", res.table())
    return re.sub(r"^best: ", "returned: ", text, flags=re.M)


def explain(a, res, d: Data) -> None:
    """The result in plain words, from the same numbers as the table."""
    ranked, tied = res.ranked, {c.name for c in res.tied}
    if not ranked:
        a.out("No model finished; the table above says why for each.", "warn")
        return
    best = res.best
    eqm = [c for c in ranked if c.uses_equation]
    dom = [c for c in ranked if not c.uses_equation]
    unit = f"in the units of {d.measured}"
    a.out("")
    a.out("What this means", "title")
    if not eqm:
        a.out("  None of the models that use your equation finished, so this run cannot say whether "
              "it helps. The table says why for each.")
    elif not dom:
        a.out(f"  Only models that use the equation finished. Returned: {best.title}.")
    else:
        e, n = eqm[0], dom[0]
        tied_eq = [c for c in res.tied if c.uses_equation]
        tied_do = [c for c in res.tied if not c.uses_equation]
        if len(res.tied) > 1 and tied_eq and tied_do:
            e, n = tied_eq[0], tied_do[0]
            a.out(f"  This data cannot tell whether your equation helps: {_short(e)}, which uses it "
                  f"(off by {_fmt(e.holdout_rmse)}), and {_short(n)}, which does not (off by "
                  f"{_fmt(n.holdout_rmse)}), are within the noise of each other on the held-out "
                  f"points (both marked = in the table). Returned: {best.title}, the one the "
                  f"equation constrains most.")
        elif e.holdout_rmse < n.holdout_rmse:
            r = n.holdout_rmse / e.holdout_rmse
            a.out(f"  Your equation helped. On points the models never saw, the most accurate model "
                  f"using it ({_short(e)}) was off by {_fmt(e.holdout_rmse)} {unit}, and the most "
                  f"accurate data-only model ({_short(n)}) by {_fmt(n.holdout_rmse)}"
                  + (f", {math.floor(r * 10) / 10:.1f} times as much" if r >= 1.1
                     else ", slightly more") +
                  f". Returned: {best.title}.")
        else:
            a.out(f"  Your equation did not help on this data. On points the models never saw, the "
                  f"most accurate data-only model ({_short(n)}) was off by {_fmt(n.holdout_rmse)} {unit}, "
                  f"and the most accurate model using the equation ({_short(e)}) by "
                  f"{_fmt(e.holdout_rmse)}. "
                  f"Possible reasons: the equation does not describe this data, the data is too "
                  f"small or noisy to show it, or the fits did not settle (read any notes above). "
                  f"Constants from an equation that does not describe the data mean little, however "
                  f"small their +-. Returned: {best.title}.")
    ef = res.candidates.get("equation_fit")
    if ef is not None and ef.ok and ef.params:
        vals = ", ".join(_with_se(k, v, ef.param_se.get(k)) for k, v in ef.params.items())
        on = (f"fitted on the {res.n_train} training points" if res.folds == 1 else
              f"refitted on all {res.n_train} points")
        a.out(f"  Constants from the classical fit, {on}: {vals}. The +- is one standard error "
              f"(roughly a 68% range), if the equation is exactly right. Units are your columns': "
              f"a rate is per one unit of {d.time}.")
    elif eqm and eqm[0].params:
        vals = ", ".join(f"{k}={v:.4g}" for k, v in eqm[0].params.items())
        a.out(f"  Constants from {eqm[0].title}: {vals} (no standard errors: only the classical "
              f"fit gives them).")
    if eqm:
        poor = [c for c in eqm if c.params and c.holdout_rmse > 3 * eqm[0].holdout_rmse]
        if poor:
            k = int(min(c.holdout_rmse for c in poor) / eqm[0].holdout_rmse)      # rounded down
            a.out(f"  Ignore the constants of {' and '.join(_short(c) for c in poor)}: on the held-out "
                  f"points {'it was' if len(poor) == 1 else 'they were'} off by {k} or more times as "
                  f"much as {_short(eqm[0])}.", "warn")
    if res.constant_notes():
        a.out("  Some constants are not pinned down by this data (the 'not pinned down' lines "
              "above). Do not trust their values; if you know one, run again and give it.", "warn")
    if res.disagreement_notes():
        a.out("  The equation models disagree on a constant (the 'models disagree' line above).", "warn")
    if res.folds == 1 and res.n_holdout < 10:
        a.out(f"  Only {res.n_holdout} points were held back for testing, so treat the ranking as a "
              f"rough guide. " + ("'Careful' scoring tests every point." if res.split == "random" else
                                 "'Careful' scoring averages over several forecast windows."), "warn")
    if res.split == "forecast" and d.position and any(c.name == "equation_fit" for c in ranked):
        a.out("  Forecasting with position: past the last training time the classical fit holds "
              "the edge values at their last fitted values; they are not predicted.", "dim")


def save_data_copy(d: Data, folder: Path) -> Path | None:
    """The data with the columns named as the scripts expect (x, t, u). None when the file
    already uses those names."""
    if not d.renamed and d.source != "example":
        return None
    out = folder / ("data.csv" if d.source == "example" else Path(d.source).stem + "_equationnet.csv")
    cols = d.columns()
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(cols))
        for row in zip(*cols.values()):
            w.writerow(["" if not np.isfinite(v) else repr(float(v)) for v in row])
    return out


def ask_numbers(a, msg: str) -> list[float]:
    def check(text):
        try:
            vals = [float(p) for p in text.replace(";", ",").split(",") if p.strip()]
        except ValueError:
            return "type numbers separated by commas, e.g. 12, 14, 16"
        if not vals:
            return "type at least one number"
        return None if all(np.isfinite(vals)) else "type finite numbers"
    return [float(p) for p in a.text(msg, validate=check).replace(";", ",").split(",") if p.strip()]


def predict(a, res, d: Data) -> None:
    best = res.best
    t = ask_numbers(a, f"{d.time} values to predict at (comma-separated):")
    if d.position:
        x = ask_numbers(a, f"{d.position} values (every pair with the {d.time} values is used):")
        pts = np.array([[xi, ti] for ti in t for xi in x])
    else:
        pts = np.array([[ti] for ti in t])
    vals = best.predict(pts)
    a.out(f"Predictions of {best.title}:", "title")
    head = "  " + "  ".join(f"{d.name(k):>12}" for k in d.indep) + f"  {d.measured:>12}"
    a.out(head)
    for row, v in zip(pts, vals):
        a.out("  " + "  ".join(f"{q:>12.6g}" for q in row) + f"  {v:>12.6g}")
    tmin, tmax = np.nanmin(d.cols[d.time]), np.nanmax(d.cols[d.time])
    outside = [v for v in t if v < tmin or v > tmax]
    if outside:
        a.out(f"  {', '.join(f'{v:g}' for v in outside)} {'is' if len(outside) == 1 else 'are'} outside "
              f"your data's {d.time} range ({tmin:g} to {tmax:g}). Models that use the equation carry it "
              f"forward there; data-only models only guess.", "warn")
    a.out("  No uncertainty range is given for predictions in this version.", "dim")
    if d.position and best.name == "equation_fit" and np.max(t) > tmax:
        a.out(f"  After {d.time} = {tmax:g} the classical fit holds the edge values at their last "
              "fitted values; those predictions are not from the equation.", "warn")
    if a.confirm("Save these to a CSV file?", default=False):
        out = Path(a.text("File name:", default="predictions.csv"))
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([d.name(k) for k in d.indep] + [d.measured])
            for row, v in zip(pts, vals):
                w.writerow([*map(float, row), float(v)])
        a.out(f"saved {out.resolve()}", "good")


def save_scripts(a, res, d: Data, which: str) -> None:
    folder = Path(a.text("Folder to save into:", default="equationnet_output")).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    if which == "best":
        paths = [res.best.save(folder / f"{res.best.name}.py")]
    else:
        paths = res.save_scripts(folder)
    copy = save_data_copy(d, folder)
    for p in paths:
        a.out(f"saved {p.resolve()}", "good")
    data_arg = copy.name if copy else Path(d.source).name
    if copy:
        a.out(f"saved {copy.resolve()}  (your data with the columns named "
              f"{', '.join(d.letters)}, as the scripts expect)", "good")
    n_all = len(next(iter(d.columns().values())))
    a.out(f"Each script trains its model again on all the data you give it ({n_all} rows here), so "
          + (f"its constants can differ a little from the table's, which came from the {res.n_train} "
             f"training points. " if res.folds == 1 else "") +
          "It needs only numpy, scipy and (for the networks) torch.", "dim")
    a.out("To run it later, open a terminal in that folder and type:", "dim")
    a.out(f"  python {paths[0].name} {data_arg}")
    a.out(f"(python here is the one that runs equationnet: {sys.executable})", "dim")


def run(asker=None, models="all") -> int:
    """The guided session. Returns the exit code."""
    a = asker or default_asker()
    try:
        a.out(f"equationnet {VERSION}: is your equation better than no equation?", "title")
        a.out("You give it data and a differential equation. It trains three models that use the "
              "equation and three that use only the data, tests all six on points they never saw, "
              "and ranks them. Esc or Ctrl-C leaves at any time.", "dim")
        a.out("")
        while True:                                     # one data set per pass
            d = ask_data(a)
            show_data(a, d)
            ask_roles(a, d)
            c = Choice()
            while True:                                 # one equation per pass
                c.equation = ask_equation(a, d)
                c.known = ask_known(a, d, c.equation)
                c.split, c.cv = ask_test(a, d)
                a.out("")
                a.out("Training six models; each prints a line when it finishes.", "dim")
                try:
                    try:
                        res = fit(c.equation, d.columns(), target="u", known=c.known, models=models,
                                  split=c.split, cv=c.cv, verbose=True)
                    except ValueError as exc:
                        if c.cv == 1 or not re.search(r"fold|origin", str(exc)):
                            raise
                        a.out(f"'Careful' scoring needs more distinct {d.time} values than this data "
                              f"has ({exc}). Running Standard instead.", "warn")
                        c.cv = 1
                        res = fit(c.equation, d.columns(), target="u", known=c.known, models=models,
                                  split=c.split, cv=1, verbose=True)
                except (EquationError, ValueError, NotApplicable) as exc:
                    a.out(f"That did not run: {exc}", "warn")
                    a.out("Pick again (the data stays loaded).", "dim")
                    continue
                a.out("")
                a.out(guided_table(res))
                a.out(LEGEND, "dim")
                explain(a, res, d)
                a.out("")
                nxt = after(a, res, d)
                if nxt == "data":
                    break
        return 0
    except Quit:
        a.out("")
        a.out("Bye.")
        return 0


def after(a, res, d: Data) -> str:
    if not res.ranked:                      # nothing to save or predict with
        pick = a.select("What next?", [("Try another equation on the same data", "equation"),
                                       ("Load other data", "data"), ("Quit", "quit")])
        if pick == "quit":
            raise Quit
        return pick
    while True:
        pick = a.select("What next?", [
            ("Try another equation on the same data", "equation"),
            (f"Save the returned model ({res.best.title}) as a Python script", "best"),
            ("Save all the models as Python scripts", "all"),
            (f"Predict {d.measured} at new {d.time} values", "predict"),
            ("What do these words mean?", "words"),
            ("Load other data", "data"),
            ("Quit", "quit")])
        if pick == "words":
            a.out(GLOSSARY)
            continue
        if pick == "quit":
            raise Quit
        if pick in ("equation", "data"):
            return pick
        if pick == "predict":
            predict(a, res, d)
        else:
            save_scripts(a, res, d, pick)
        a.out("")
