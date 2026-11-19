"""The guided session (`equationnet` with no arguments), stage I in TESTPLAN.md.

These are the fast checks (a few models, small data). bench/stages/stage_i.py runs every
starter equation with all six models."""

import subprocess
import sys

import numpy as np
import pytest

from equationnet import cli, guide
from equationnet.fit import Candidate, Result
from equationnet.parse import parse_equation


class Scripted:
    """An asker that answers from a list. A validated question takes answers until one
    passes, recording each rejection, as a person would retype."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.lines, self.rejected, self.asked = [], [], []

    def out(self, text="", style=None):
        self.lines.append(text)

    def _next(self, msg):
        self.asked.append(msg)
        if not self.answers:
            raise guide.Quit
        a = self.answers.pop(0)
        if a is guide.Quit:
            raise guide.Quit
        return a

    def _validated(self, msg, validate):
        while True:
            a = self._next(msg)
            err = validate(a.strip()) if validate else None
            if err is None:
                return a.strip()
            self.rejected.append((msg, a, err))

    def select(self, msg, choices, default=None):
        a = self._next(msg)
        values = [v for _, v in choices]
        if a == "default":
            assert default in values
            return default
        assert a in values, f"{a!r} is not offered for {msg!r}: {values}"
        return a

    def text(self, msg, default="", validate=None):
        return self._validated(msg, validate)

    def confirm(self, msg, default=True):
        return self._next(msg)

    def checkbox(self, msg, choices):
        a = self._next(msg)
        assert set(a) <= {v for _, v in choices}
        return a

    def path(self, msg, validate=None):
        return self._validated(msg, validate)

    @property
    def text_out(self):
        return "\n".join(self.lines)


def decay_csv(path, n=20, k=0.7, noise=0.01, seed=900):
    """Real-world column names: units in brackets, spaces, plus a column not to use."""
    t = np.linspace(0, 4, n)
    u = 2.0 * np.exp(-k * t) + noise * np.random.default_rng(seed).standard_normal(n)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("time (s),signal [mV],operator note\n")
        for a, b in zip(t, u):
            fh.write(f"{float(a)!r},{float(b)!r},1\n")
    return path


FAST = ["poly", "spline", "equation_fit"]


def test_renamed_columns_end_to_end(tmp_path, monkeypatch):
    """I3: columns named 'time (s)' and 'signal [mV]' run; the saved script runs on the
    saved data copy and finds the same k."""
    monkeypatch.chdir(tmp_path)
    csv = decay_csv(tmp_path / "my data.csv")
    a = Scripted(["file", f'"{csv}"',            # dragged-in paths come quoted
                  "signal [mV]", "time (s)", None,
                  "u_t = -k*u", False, "random", 1,
                  "best", "out",
                  "quit"])
    assert guide.run(a, models=FAST) == 0
    out = a.text_out
    assert "u = signal [mV]" in out and "t = time (s)" in out and "not used: operator note" in out
    assert "What this means" in out and "Your equation helped" in out
    assert "Bye." in out
    script = tmp_path / "out" / "equation_fit.py"
    copy = tmp_path / "out" / "my data_equationnet.csv"
    assert script.exists() and copy.exists()
    assert copy.read_text().splitlines()[0] == "t,u"
    run = subprocess.run([sys.executable, "-I", str(script), copy.name], cwd=script.parent,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    k = float(run.stdout.split("'k': ")[1].split("}")[0].split(",")[0])
    assert abs(k - 0.7) < 0.05


def test_example_data_and_another_equation(tmp_path, monkeypatch):
    """The packaged example loads; 'Try another equation' keeps the data and asks again."""
    monkeypatch.chdir(tmp_path)
    a = Scripted(["example", "u", "t",
                  "u_t = -k*u", False, "random", 1,
                  "equation", "own", "u_t = -k*u + b", False, "forecast", 1,
                  "quit"])
    assert guide.run(a, models=["poly", "spline"]) == 0
    out = a.text_out
    assert guide.EXAMPLE_NOTE in out
    assert out.count("What this means") == 2
    assert "split: forecast" in out


def test_wrong_answers_are_said_in_words_and_asked_again(tmp_path, monkeypatch):
    """I4: every wrong answer gets a message and the question again; no traceback."""
    monkeypatch.chdir(tmp_path)
    csv = decay_csv(tmp_path / "d.csv")
    (tmp_path / "book.xlsx").write_bytes(b"PK\x03\x04")
    (tmp_path / "one.csv").write_text("t\n1\n2\n3\n4\n5\n6\n")
    (tmp_path / "words.csv").write_text("t,u\n1,a\n2,b\n")
    (tmp_path / "semi.csv").write_text("t;u\n1;2\n")
    (tmp_path / "blob.csv").write_bytes(bytes(range(256)) * 4)
    a = Scripted(["file", "", "nope.csv", str(tmp_path), str(tmp_path / "book.xlsx"),
                  str(tmp_path / "one.csv"), str(tmp_path / "words.csv"), str(tmp_path / "semi.csv"),
                  str(tmp_path / "blob.csv"), str(csv),
                  "signal [mV]", "time (s)", None,
                  "help", "own", "u = k", "u_t = k*", "u_x = -k*u", "u_t = -k*u",
                  True, ["k"], "four", "inf", "0.7",
                  "random", 1, guide.Quit])
    assert guide.run(a, models=FAST) == 0
    msgs = [err for _, _, err in a.rejected]
    assert len(msgs) == 13, msgs
    assert all(isinstance(m, str) and m for m in msgs)
    joined = " ".join(msgs)
    for words in ("drag the file", "no such file", "is a folder", "Save As", "one column",
                  "'u' is not used (text", "semicolons", "not a text file", "type a number", "finite"):
        assert words in joined, words
    assert "How to write an equation" in a.text_out
    assert "Traceback" not in a.text_out
    assert "k=0.7" in a.text_out.replace(" ", "")          # the known value was used


def test_quit_anywhere_is_clean():
    for n in range(4):
        a = Scripted(["example", "u", "t", "u_t = -k*u"][:n] + [guide.Quit])
        assert guide.run(a) == 0
        assert a.lines[-1] == "Bye."


def _result(eq_err, data_err, n=8, seed=0):
    """A Result with one equation model and one data-only model whose held-out
    residuals have the given RMS sizes."""
    rng = np.random.default_rng(seed)
    eq = parse_equation("u_t = -k*u")

    def cand(name, title, uses, size):
        r = rng.standard_normal(n)
        r = size * r / np.sqrt(np.mean(r ** 2))
        return Candidate(name, title, uses, status="ok", holdout_rmse=float(size),
                         train_rmse=0.005,                 # a small measurement noise
                         fold_rmse=[float(size)], fold_errors=[r],
                         params={"k": 0.7} if uses else {}, param_se={"k": 0.01} if uses else {})

    cands = {"equation_fit": cand("equation_fit", "Equation fit (classical)", True, eq_err),
             "spline": cand("spline", "Smoothing spline (data only)", False, data_err)}
    return Result(equation=eq, split="random", n_train=32, n_holdout=n, candidates=cands)


@pytest.mark.parametrize("eq_err, data_err, phrase", [
    (0.01, 1.0, "Your equation helped"),
    (1.0, 0.01, "Your equation did not help"),
    (0.100, 0.101, "cannot tell whether your equation helps"),
])
def test_summary_matches_the_ranking(eq_err, data_err, phrase):
    """I2: the plain-words summary agrees with the ranking and the ties."""
    res = _result(eq_err, data_err)
    a = Scripted([])
    d = guide.Data("x.csv", {"t": np.zeros(3), "u": np.zeros(3)}, "u", "t")
    guide.explain(a, res, d)
    out = a.text_out
    assert phrase in out
    assert f"Returned: {res.best.title}" in out
    others = {"Your equation helped", "Your equation did not help", "cannot tell whether your equation helps"} - {phrase}
    assert not any(o in out for o in others)


def test_plain_asker_numbered_menus():
    """I5 (in-process part): the fallback reads numbers, defaults, and re-asks."""
    answers = iter(["9", "x", "2", "", "maybe", "y", "1, 3", "", "abc"])
    shown = []
    p = guide.PlainAsker(read=lambda prompt: next(answers), write=shown.append)
    assert p.select("pick", [("a", 1), ("b", 2)]) == 2
    assert p.select("pick", [("a", 1), ("b", 2)], default=1) == 1
    assert p.confirm("ok?") is True
    assert p.checkbox("which", [("a", "A"), ("b", "B"), ("c", "C")]) == ["A", "C"]
    assert p.text("name", default="d") == "d"
    assert p.text("t", validate=lambda s: None if s == "abc" else "no") == "abc"
    assert sum("type a number from 1 to 2" in s for s in shown) == 2
    eof = guide.PlainAsker(read=lambda prompt: (_ for _ in ()).throw(EOFError), write=shown.append)
    with pytest.raises(guide.Quit):
        eof.select("pick", [("a", 1)])


def test_piped_session_runs_with_numbered_menus(tmp_path):
    """I5: no terminal -> numbered menus; the installed command runs end to end."""
    csv = decay_csv(tmp_path / "d.csv")
    answers = "\n".join(["1", str(csv), "2", "1", "1", "1", "n", "1", "1", "6"]) + "\n"
    run = subprocess.run([sys.executable, "-c", "import sys; from equationnet.cli import main; "
                          "sys.exit(main([]))"], input=answers, capture_output=True, text=True,
                         timeout=900, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "  1. A CSV file on this computer" in run.stdout
    assert "What this means" in run.stdout and "Bye." in run.stdout
    assert "Traceback" not in run.stdout + run.stderr


def test_cli_without_arguments_starts_the_guide(monkeypatch):
    called = []
    monkeypatch.setattr(guide, "run", lambda: called.append(1) or 0)
    assert cli.main([]) == 0 and cli.main(["start"]) == 0
    assert called == [1, 1]


@pytest.mark.parametrize("typed, path", [
    ('"C:\\data\\my file.csv"', "C:\\data\\my file.csv"),
    ("'/home/me/a b.csv' ", "/home/me/a b.csv"),
    ("& 'C:\\x.csv'", "C:\\x.csv"),
    ("plain.csv", "plain.csv"),
])
def test_dragged_paths(typed, path):
    assert guide.clean_path(typed) == path


def _drive(keys, fn):
    """Run a MenuAsker question on scripted keystrokes (no real terminal)."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            try:
                return fn(guide.MenuAsker())
            except guide.Quit:
                return "QUIT"


def test_arrow_key_menus():
    """The terminal askers: arrows, defaults, validation, ticks, and Ctrl-C / Esc to leave."""
    three = [("a", 1), ("b", 2), ("c", 3)]
    assert _drive("\x1b[B\x1b[B\r", lambda a: a.select("pick", three)) == 3
    assert _drive("\r", lambda a: a.select("pick", three, default=2)) == 2
    assert _drive("bad\r\x08\x08\x08ok\r",
                  lambda a: a.text("t", validate=lambda s: None if s == "ok" else "no")) == "ok"
    assert _drive(" \x1b[B\x1b[B \r", lambda a: a.checkbox("c", [("a", "A"), ("b", "B"), ("c", "C")])) == ["A", "C"]
    assert _drive("y", lambda a: a.confirm("ok?", default=False)) is True
    assert _drive("\x03", lambda a: a.select("pick", three)) == "QUIT"
    assert _drive("\x1b", lambda a: a.text("t")) == "QUIT"


# ---- fixes from the stage I7 fresh-user review (2026-11-19)

def test_column_guesses_on_a_spreadsheet_export():
    cols = {"Minutes elapsed": np.arange(0, 93, 3.0),
            "Coffee temp (F)": 72 + 118 * np.exp(-0.035 * np.arange(0, 93, 3.0)),
            "Room temp (F)": 72 + 0.3 * np.sin(np.arange(31.0))}
    t = guide._time_guess(cols)
    assert t == "Minutes elapsed"
    assert guide._measured_guess(cols, t) == "Coffee temp (F)"


def test_brackets_after_a_name_are_confirmed_not_silently_learned(tmp_path, monkeypatch):
    """u_t = k(L-u) reads k(...) as an unknown function: the session says so and asks."""
    monkeypatch.chdir(tmp_path)
    csv = decay_csv(tmp_path / "d.csv")
    a = Scripted(["file", str(csv), "signal [mV]", "time (s)", None,
                  "own", "T_t = -k*T", "u_t = k(L-u)", False,      # wrong letter, then k( ... ) declined
                  "u_t = -k*u", guide.Quit])
    guide.run(a, models=FAST)
    assert any("is called u" in err for _, _, err in a.rejected)
    assert any("unknown function" in q and "k*(...)" in q for q in a.asked)


def test_careful_forecast_falls_back_to_standard_and_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = decay_csv(tmp_path / "d.csv")
    real, calls = guide.fit, []

    def fake(*args, **kw):
        calls.append(kw["cv"])
        if kw["cv"] > 1:
            raise ValueError("cv=3 forecast origins of 4 time(s) each need more than 20 distinct times")
        return real(*args, **kw)

    monkeypatch.setattr(guide, "fit", fake)
    a = Scripted(["file", str(csv), "signal [mV]", "time (s)", None,
                  "u_t = -k*u", False, "forecast", 3, "quit"])
    assert guide.run(a, models=FAST) == 0
    assert calls == [3, 1]
    assert "Running Standard instead" in a.text_out and "What this means" in a.text_out


def test_python_advice_is_put_in_session_terms():
    class R:
        def table(self):
            return "not pinned down: k: ... If you know it, pass known={'k': ...}. Otherwise"
    assert guide.guided_table(R()) == ("not pinned down: k: ... If you know it, give k when asked "
                                       "which constants you already know. Otherwise")


def test_prediction_outside_the_data_is_flagged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = decay_csv(tmp_path / "d.csv")
    a = Scripted(["file", str(csv), "signal [mV]", "time (s)", None,
                  "u_t = -k*u", False, "random", 1,
                  "predict", "two hours", "1, 9", False, "words", "quit"])
    assert guide.run(a, models=FAST) == 0
    out = a.text_out
    assert "9 is outside your data's time (s) range (0 to 4)" in out
    assert "No uncertainty range" in out and "Words used here" in out
    assert any("type numbers separated by commas" in err for _, _, err in a.rejected)


def test_summary_uses_the_tied_set_when_folds_reorder_models():
    """With cross-validation the ranking (mean fold error) and the tie test (pooled
    squared errors) can disagree; the summary follows the '=' marks in the table."""
    rng = np.random.default_rng(1)
    eq = parse_equation("u_t = -k*u")

    def cand(name, title, uses, rank_rmse, err_size):
        r = rng.standard_normal(8)
        r = err_size * r / np.sqrt(np.mean(r ** 2))
        return Candidate(name, title, uses, status="ok", holdout_rmse=rank_rmse, train_rmse=0.005,
                         fold_rmse=[rank_rmse], fold_errors=[r])

    cands = {"equation_fit": cand("equation_fit", "Equation fit (classical)", True, 0.100, 0.100),
             "nn": cand("nn", "Neural network (data only)", False, 0.105, 1.0),       # ranked 2nd, not tied
             "spline": cand("spline", "Smoothing spline (data only)", False, 0.20, 0.1005)}   # tied
    res = Result(equation=eq, split="random", n_train=32, n_holdout=8, candidates=cands)
    assert {c.name for c in res.tied} == {"equation_fit", "spline"}
    a = Scripted([])
    guide.explain(a, res, guide.Data("x.csv", {"t": np.zeros(3), "u": np.zeros(3)}, "u", "t"))
    assert "cannot tell whether your equation helps" in a.text_out
    assert "the smoothing spline, which does not" in a.text_out
    assert "most accurate data-only model" not in a.text_out


def test_poor_classical_fit_constants_are_flagged_too():
    res = _result(0.01, 1.0)
    res.candidates["pinn"] = Candidate("pinn", "PINN (neural network + equation)", True, status="ok",
                                       holdout_rmse=0.001, train_rmse=0.005, fold_rmse=[0.001],
                                       fold_errors=[np.full(8, 0.001)], params={"k": 0.7})
    a = Scripted([])
    guide.explain(a, res, guide.Data("x.csv", {"t": np.zeros(3), "u": np.zeros(3)}, "u", "t"))
    assert "Ignore the constants of the classical fit" in a.text_out
    assert "off by 10 or more times" in a.text_out
