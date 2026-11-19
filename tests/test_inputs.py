"""Stage B (bad and odd input) and stage E (security) of TESTPLAN.md.

Each case states its expected behaviour BEFORE it was first run (2026-11-16):
  OK      it works (and, where it matters, gives the same answer as the clean input)
  STOP    it stops with an error message in words that names the problem
  SAY     it works and the result says in words what was changed or left out
Nothing in the input may be dropped or reinterpreted silently (B2), and no
injected code may run or reach a generated script (E1).
"""

import ast
import subprocess
import sys
import time

import numpy as np
import pytest

import equationnet as en
from equationnet.models import NotApplicable

EQ = "u_t = -k*u"


def decay(n=30, k=0.7, noise=0.01, seed=0):
    t = np.linspace(0, 4, n)
    return t, 2 * np.exp(-k * t) + noise * np.random.default_rng(seed).standard_normal(n)


def fit(data, eq=EQ, **kw):
    kw.setdefault("models", ["equation_fit"])
    kw.setdefault("verbose", False)
    return en.fit(eq, data, **kw)


def stops(match, call, *a, **kw):
    """STOP: raises an error whose message contains `match` (case-insensitive)."""
    with pytest.raises(Exception) as info:
        call(*a, **kw)
    assert match.lower() in str(info.value).lower(), str(info.value)
    assert not isinstance(info.value, (KeyError, IndexError, AttributeError, TypeError, ZeroDivisionError)), \
        f"a bare {type(info.value).__name__} is not a message in words: {info.value}"


def csv_file(tmp_path, text, name="d.csv", encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


def clean_csv(n=30):
    t, u = decay(n)
    return "t,u\n" + "".join(f"{a},{b}\n" for a, b in zip(t, u))


# ---------------------------------------------------------------- stage B: data

def test_b01_unsorted_times_give_the_same_answer():                       # OK
    t, u = decay()
    order = np.random.default_rng(1).permutation(len(t))
    a = fit({"t": t, "u": u}).best.params["k"]
    b = fit({"t": t[order], "u": u[order]}).best.params["k"]
    assert abs(a - 0.7) < 0.02 and abs(b - 0.7) < 0.02


def test_b02_repeated_times_work():                                       # OK
    t, u = decay(20)
    t2 = np.concatenate([t, t])
    u2 = np.concatenate([u, u + 0.01 * np.random.default_rng(2).standard_normal(len(u))])
    assert abs(fit({"t": t2, "u": u2}).best.params["k"] - 0.7) < 0.03


def test_b03_missing_values_are_left_out_and_said(tmp_path):              # SAY
    text = clean_csv().splitlines()
    text[5] = text[5].split(",")[0] + ","          # empty u
    text[9] = text[9].split(",")[0] + ",nan"
    r = fit(csv_file(tmp_path, "\n".join(text) + "\n"))
    assert r.n_train + r.n_holdout == 28
    assert "2 rows" in r.table() and "missing" in r.table().lower()


def test_b04_text_in_a_number_column_stops(tmp_path):                     # STOP
    text = clean_csv().splitlines()
    text[3] = text[3].split(",")[0] + ",abc"
    stops("abc", fit, csv_file(tmp_path, "\n".join(text) + "\n"))


def test_b05_missing_target_column_stops():                               # STOP
    t, u = decay()
    stops("no column 'u'", fit, {"t": t, "y": u})


def test_b06_no_variable_column_stops():                                  # STOP
    stops("variable", fit, {"u": decay()[1]})


def test_b07_extra_column_not_in_the_equation_stops():                    # STOP
    t, u = decay()
    stops("notes", fit, {"t": t, "notes": np.arange(len(t)), "u": u})


def test_b08_constant_data_does_not_crash():                              # OK or STOP in words
    t = np.linspace(0, 4, 30)
    try:
        r = fit({"t": t, "u": np.full(30, 3.0)}, models="all",
                config={"pinn": {"steps": 50, "lbfgs": 5}, "poly_physics": {"steps": 50, "lbfgs": 5},
                        "nn": {"steps": 50, "lbfgs": 5}})
    except Exception as exc:
        assert not isinstance(exc, (KeyError, IndexError, ZeroDivisionError)), exc
        return
    r.table()


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_b09_too_few_points_stops(n):                                     # STOP
    t, u = decay(n)
    stops("points", fit, {"t": t, "u": u})


def test_b11_a_large_file_finishes():                                     # OK, under 3 minutes
    t, u = decay(100_000, noise=0.01)
    t0 = time.perf_counter()
    r = fit({"t": t, "u": u}, models=["equation_fit", "spline", "poly"])
    assert time.perf_counter() - t0 < 180
    assert abs(r["equation_fit"].params["k"] - 0.7) < 0.01


def test_b12_negative_and_huge_times_work():                              # OK
    t, u = decay()
    r1 = fit({"t": t * 1e6 - 3e6, "u": u})
    assert abs(r1.best.params["k"] * 1e6 - 0.7) < 0.03


@pytest.mark.parametrize("name", ["sin", "pi", "exp"])
def test_b13_column_named_like_a_math_name_stops(name):                   # STOP
    t, u = decay()
    stops(name, fit, {name: t, "u": u}, eq=f"u_{name} = -k*u" if len(name) == 1 else "u_t = -k*u")


def test_b13b_column_named_E_or_I_works():                                # OK (sympy constants)
    t, u = decay()
    for name in ("E", "I"):
        r = fit({name: t, "u": u}, eq=f"u_{name} = -k*u")
        assert abs(r.best.params["k"] - 0.7) < 0.02


@pytest.mark.parametrize("eq,match", [
    ("u_t = -k*", "could not read"),
    ("u_t -k*u", "one '='"),
    ("u_x = -k*u", "variable"),
    ("v_t = -k*v", "derivative"),
    ("u_t = -k*u = 0", "one '='"),
])
def test_b14_bad_equations_stop_in_words(eq, match):                      # STOP
    t, u = decay()
    stops(match, fit, {"t": t, "u": u}, eq=eq)


def test_b17_unicode_symbols_stop_and_name_the_character():               # STOP
    t, u = decay()
    stops("−", fit, {"t": t, "u": u}, eq="u_t = −k*u")


def test_b18_windows_line_endings(tmp_path):                              # OK
    r = fit(csv_file(tmp_path, clean_csv().replace("\n", "\r\n")))
    assert abs(r.best.params["k"] - 0.7) < 0.02


def test_b19_excel_byte_order_mark(tmp_path):                             # OK
    r = fit(csv_file(tmp_path, "﻿" + clean_csv()))
    assert abs(r.best.params["k"] - 0.7) < 0.02


def test_b20_blank_lines_and_trailing_spaces(tmp_path):                   # OK
    text = clean_csv().replace("\n", " \n", 3) + "\n\n"
    r = fit(csv_file(tmp_path, text))
    assert abs(r.best.params["k"] - 0.7) < 0.02


def test_b21_semicolon_separated_file_stops(tmp_path):                    # STOP
    stops("comma", fit, csv_file(tmp_path, clean_csv().replace(",", ";")))


def test_b22_header_only_stops(tmp_path):                                 # STOP
    stops("points", fit, csv_file(tmp_path, "t,u\n"))


def test_b23_columns_of_different_lengths_stop():                         # STOP
    t, u = decay()
    stops("length", fit, {"t": t, "u": u[:-3]})


def test_b24_dataframe_input():                                           # OK
    pd = pytest.importorskip("pandas")
    t, u = decay()
    assert abs(fit(pd.DataFrame({"t": t, "u": u})).best.params["k"] - 0.7) < 0.02


def test_b26_known_value_that_is_not_a_number_stops():                    # STOP
    t, u = decay()
    stops("'A'", fit, {"t": t, "u": u}, eq="u_t = -k*(u - A)", known={"A": "zero"})


def test_b28_other_target_name():                                         # OK
    t, u = decay()
    r = fit({"t": t, "T": u}, eq="T_t = -k*T", target="T")
    assert abs(r.best.params["k"] - 0.7) < 0.02


@pytest.mark.parametrize("name", ["np", "torch", "math", "V", "P", "F", "residual"])
def test_b29_constants_named_like_generated_code_names(name):             # OK
    t, u = decay()
    eq = (f"u_t = -{name}*exp(-c*t)*u" if name != "f_0" else "u_t = -f_0*u + 0.001*f(u)")
    r = fit({"t": t, "u": u}, eq=eq, models=["equation_fit", "pinn"],
            config={"pinn": {"steps": 50, "lbfgs": 5}})
    for m in ("equation_fit", "pinn"):
        assert r[m].ok, (m, r[m].status)
    assert abs(r["equation_fit"].params[name] - 0.7) < 0.05


def test_b30_constant_named_like_an_unknown_functions_internal_name():   # OK, no clash
    from equationnet.parse import parse_equation
    eq = parse_equation("u_t = -f_0*u + f(u)")
    assert eq.params == ["f_0"] and [u.placeholder for u in eq.ufuncs] == ["_f_0"]
    assert {"f_0", "_f_0"} <= {str(s) for s in eq.residual.free_symbols}
    t, u = decay()        # f(u) can absorb -f_0*u, so f_0 is not determined: it must be flagged
    r = fit({"t": t, "u": u}, eq="u_t = -f_0*u + f(u)")
    assert r["equation_fit"].ok and "not pinned down: f_0" in r.table()


@pytest.mark.parametrize("holdout", [0, 1, 1.5, -0.2])
def test_b31_holdout_out_of_range_stops(holdout):                         # STOP
    t, u = decay()
    stops("holdout", fit, {"t": t, "u": u}, holdout=holdout)


def test_b33_unknown_model_name_stops():                                  # STOP
    t, u = decay()
    stops("pinnn", fit, {"t": t, "u": u}, models=["pinnn"])


def test_b34_predict_at_with_wrong_columns_stops():                       # STOP
    t, u = decay()
    stops("column", fit, {"t": t, "u": u}, predict_at={"x": [1.0]})


def test_b36_infinite_values_are_left_out_and_said():                     # SAY
    t, u = decay()
    u[4] = np.inf
    r = fit({"t": t, "u": u})
    assert r.n_train + r.n_holdout == 29 and "1 row" in r.table()


BOOT = ("import platform; platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError()); "
        "import sys; from equationnet.cli import main; sys.exit(main(sys.argv[1:]))")


@pytest.mark.parametrize("args,match", [
    (["fit", "u_t = -k*u", "no_such_file.csv"], "no_such_file.csv"),
    (["fit", "u_t = -k*", "DATA"], "could not read"),
    (["fit", "u_t = -k*u", "DATA", "--holdout", "2"], "holdout"),
])
def test_b35_cli_errors_are_words_not_tracebacks(tmp_path, args, match):  # STOP, no traceback
    data = csv_file(tmp_path, clean_csv())
    args = [str(data) if a == "DATA" else a for a in args]
    import os
    from pathlib import Path
    src = str(Path(en.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", BOOT, *args], capture_output=True, text=True,
                         timeout=300, cwd=tmp_path, env=env)
    assert out.returncode != 0
    assert "Traceback" not in out.stderr + out.stdout, out.stderr
    assert match.lower() in (out.stderr + out.stdout).lower()


# ---------------------------------------------------------------- stage E: security

INJECTIONS = [
    'u_t = -k*u """',                                  # breaks out of the docstrings
    "u_t = -k*().__class__",                           # attribute access during parsing
    "u_t = -k*u.__class__.__mro__",
    "u_t = __import__('os').getcwd()*u",               # quotes, dunder
    "u_t = -k*u; import os",
    "u_t = -k*u\nimport os",
    "u_t = -k*u[0]",
    "u_t = -k*u \\",
    "u_t = -k*u # comment",
    "u_t = -k*{1: 2}",
    "u_t = -k*u @ u",
    "u_t = -k*u + 0*sin('x')",                         # a string literal through the prime quote
    "u_t = -k*'u'",
]


@pytest.mark.parametrize("text", INJECTIONS)
def test_e01_equation_text_cannot_inject(text):                           # STOP before parsing
    marker = []
    t, u = decay()
    stops("", fit, {"t": t, "u": u}, eq=text)
    with pytest.raises(Exception):
        en.generate(text, "equation_fit")
    assert not marker


def test_e01b_numeric_power_towers_do_not_hang():                         # STOP or OK, under 10 s
    t, u = decay()
    t0 = time.perf_counter()
    try:
        fit({"t": t, "u": u}, eq="u_t = -k*u + 0*9**9**9**9")
    except Exception:
        pass
    assert time.perf_counter() - t0 < 10


def test_e01c_very_long_equation_is_fast():                               # STOP or OK, under 10 s
    t, u = decay()
    t0 = time.perf_counter()
    try:
        fit({"t": t, "u": u}, eq="u_t = -k*u" + " + 0*u" * 3000)
    except Exception:
        pass
    assert time.perf_counter() - t0 < 10


@pytest.mark.parametrize("name", ['t"', "my var", "1t", 't"""', "t;import os", "é"])
def test_e02_column_names_cannot_inject(tmp_path, name):                  # STOP naming the column
    text = clean_csv().replace("t,u", f"{name},u", 1)
    stops(name.strip(), fit, csv_file(tmp_path, text), eq="u_t = -k*u")


@pytest.mark.parametrize("target", ['u"', "u u", "u;x"])
def test_e03_target_name_cannot_inject(target):                           # STOP
    t, u = decay()
    stops("target", fit, {"t": t, target: u}, target=target)


ACCEPTED = ["u_t = -k*u", "u_tt + c*u_t + k*u = 0", "u_t = D*u_xx", "u_t = r*u*(1 - u/K)",
            "u'' + c*u' + f(u) = 0", "u_t + u*u_x = nu*u_xx", "u_t = -k*(u - A)",
            "u_tt = -g*sin(u)", "u_t = a*exp(-b*t)*u"]


@pytest.mark.parametrize("eq", ACCEPTED)
@pytest.mark.parametrize("model", ["equation_fit", "pinn", "poly_physics", "nn", "spline", "poly"])
def test_e04_every_generated_script_parses_and_keeps_the_equation(eq, model):  # OK
    try:
        src = en.generate(eq, model)
    except Exception as exc:
        assert "not applicable" in type(exc).__name__.lower() or "classical" in str(exc) \
            or "one variable" in str(exc), exc
        return
    ast.parse(src)
    assert eq in src


# ---------------------------------------------------------------- stage C: the scripts' own reader

@pytest.mark.parametrize("case,expect", [("bom", "works"), ("extra", "works"), ("missing", "says"),
                                         ("text", "stops"), ("no_target", "stops")])
def test_c_standalone_script_reads_csv_carefully(tmp_path, case, expect):
    import os
    from pathlib import Path
    t, u = decay()
    script = tmp_path / "spline.py"
    script.write_text(en.generate(EQ, "spline"), encoding="utf-8")
    rows = [f"{a},{b}" for a, b in zip(t, u)]
    head = "t,u"
    if case == "missing":
        rows[5] = f"{t[5]},"
    if case == "text":
        rows[5] = f"{t[5]},abc"
    if case == "extra":
        head, rows = "note,u,t", [f"0,{b},{a}" for a, b in zip(t, u)]
    if case == "no_target":
        head = "t,y"
    text = ("\ufeff" if case == "bom" else "") + head + "\n" + "\n".join(rows) + "\n"
    (tmp_path / "d.csv").write_text(text, encoding="utf-8")
    boot = ("import platform; platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError()); "
            "import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    out = subprocess.run([sys.executable, "-I", "-c", boot, str(script), str(tmp_path / "d.csv")],
                         capture_output=True, text=True, timeout=300)
    text_out = out.stdout + out.stderr
    assert "Traceback" not in text_out, text_out
    if expect == "works":
        assert out.returncode == 0 and (case != "extra" or "note" in text_out)
    elif expect == "says":
        assert out.returncode == 0 and "left out 1 row" in text_out
    else:
        assert out.returncode != 0 and ("'abc'" in text_out or "'u'" in text_out)


# ---------------------------------------------------------------- stage F findings (black-box reviewer)

def test_f_ln_and_other_spellings_are_math_not_learned_functions():
    from equationnet.parse import parse_equation
    import sympy as sp
    a, b = parse_equation("u_t = -k*ln(u)"), parse_equation("u_t = -k*log(u)")
    assert not a.ufuncs and sp.simplify(a.residual - b.residual) == 0
    c = parse_equation("u_t = -k*log10(u)")
    assert not c.ufuncs and sp.simplify(c.residual - (sp.Symbol("u_t") + sp.Symbol("k") * sp.log(sp.Symbol("u")) / sp.log(10))) == 0
    for name in ("arcsin", "arctan", "erf"):
        assert not parse_equation(f"u_t = -k*{name}(u)").ufuncs


@pytest.mark.parametrize("name", ["sign", "max", "floor", "heaviside"])
def test_f_unsupported_math_names_stop_instead_of_being_learned(name):
    from equationnet.parse import parse_equation
    stops("not supported", parse_equation, f"u_t = -k*{name}(u)")


@pytest.mark.parametrize("eq,match", [
    ("u_t=-k u", "'*' missing between 'k' and 'u'"),
    ("u_t = -2u", "missing after the number"),
    ("u_t = -k*u +", "ends with '+'"),
    ("u_t = -k*(u", "brackets do not match"),
])
def test_f_syntax_errors_say_what_is_wrong(eq, match):
    from equationnet.parse import parse_equation
    stops(match, parse_equation, eq)
    with pytest.raises(Exception) as info:
        parse_equation(eq)
    assert "<unknown>" not in str(info.value)


def test_f_numbers_with_exponents_still_read():
    from equationnet.parse import parse_equation
    assert parse_equation("u_t = -1e-3*k*u").params == ["k"]


# ---------------------------------------------------------------- stage F findings (code reviewer)

def _grid(nt, nx=11, D=0.1, seed=1):
    x, t = np.linspace(0, 1, nx), np.linspace(0, 1, nt)
    XX, TT = np.meshgrid(x, t)
    U = np.exp(-D * np.pi ** 2 * TT) * np.sin(np.pi * XX)
    return {"x": XX.ravel(), "t": TT.ravel(),
            "u": U.ravel() + 0.001 * np.random.default_rng(seed).standard_normal(U.size)}


@pytest.mark.parametrize("nt", [3, 4, 10])
def test_f1_forecast_on_a_grid_holds_out_whole_time_slices(nt):
    r = fit(_grid(nt), eq="u_t = D*u_xx", split="forecast", models=["poly", "spline"])
    assert r.n_holdout >= 11 and r.n_holdout % 11 == 0
    assert all(c.ok for c in r.candidates.values()), {c.name: c.status for c in r.candidates.values()}


def test_f1_forecast_needs_two_times():
    d = _grid(2)
    d = {k: v[d["t"] == 0] for k, v in d.items()}
    stops("2 distinct times", fit, d, eq="u_t = D*u_xx", split="forecast", models=["poly"])


def test_f2_numeric_function_values_work_in_torch_models():
    t = np.linspace(0, 10, 30)
    u = np.exp(-np.log(2) / 3 * t) + 0.01 * np.random.default_rng(0).standard_normal(30)
    r = fit({"t": t, "u": u}, eq="u_t = -log(2)/h*u", models=["equation_fit", "pinn", "poly_physics"],
            config={"pinn": {"steps": 50, "lbfgs": 5}, "poly_physics": {"steps": 50, "lbfgs": 5}})
    for c in r.candidates.values():
        assert c.ok, (c.name, c.status)
    assert abs(r["equation_fit"].params["h"] - 3) < 0.1
    assert en.generate("u_t = -k*log10(u)", "pinn")      # log10 maps to log(u)/log(10)


def test_f4_pde_with_every_constant_known():
    for eq, known in (("u_t = 0.1*u_xx", None), ("u_t = D*u_xx", {"D": 0.1})):
        r = fit(_grid(8), eq=eq, known=known)
        c = r["equation_fit"]
        assert c.ok and c.holdout_rmse < 0.01, (eq, c.status, c.holdout_rmse)


def test_f5_saved_physics_script_carries_the_scored_domain():
    t = np.linspace(0, 6, 30)
    u = np.exp(-0.7 * t)
    r = fit({"t": t, "u": u}, models=["poly_physics"], predict_at={"t": [9.0]},
            config={"poly_physics": {"steps": 20, "lbfgs": 2}})
    line = [l for l in r["poly_physics"].script.splitlines() if l.startswith("DOMAIN")][0]
    assert "9.0" in line and "None" not in line.split("#")[0]


def test_f6_f7_forecast_folds_do_not_overlap_and_stop_when_too_many():
    from equationnet.fit import _folds
    t = np.linspace(0, 10, 60)
    folds = _folds(t[:, None], t, ["t"], "forecast", 0.2, 0, 3)
    pooled = np.concatenate([np.flatnonzero(te) for _, te in folds])
    assert len(pooled) == len(set(pooled.tolist()))
    stops("forecast origins", _folds, t[:, None], t, ["t"], "forecast", 0.2, 0, 10)


def test_f8_primes_with_a_long_variable_name_stop_clearly():
    from equationnet.parse import parse_equation
    stops("one-letter variable name", parse_equation, "u'' + k*u = 0", indep=["time"])


def test_f9_time_is_the_first_order_variable():
    g = _grid(8)
    d = {"s": g["t"], "x": g["x"], "u": g["u"]}
    r = fit(d, eq="u_s = D*u_xx")
    assert r.equation.time == "s" and r["equation_fit"].ok, r["equation_fit"].status


def test_f10_space_before_a_bracket_is_not_a_function_call():
    from equationnet.parse import parse_equation
    stops("'*' missing", parse_equation, "u_t = -k (u - A)")


def test_f11_spline_says_why_with_too_few_points():
    t = np.linspace(0, 1, 5)
    r = fit({"t": t, "u": np.exp(-t)}, models=["spline", "poly"])
    s = r["spline"].status
    assert s == "ok" or "at least 5 distinct" in s


def test_f12_settings_are_checked():
    stops("whole number", en.generate, EQ, "pinn", config={"steps": "__import__('os')"})
    stops("whole number", en.generate, EQ, "pinn", config={"steps": -1})
    stops("would not be used", en.generate, EQ, "pinn", config={"pinn": {"steps": 7}, "nn": {"bogus": 1}})


def test_f13_polynomial_on_two_time_slices_is_not_a_constant():
    g = _grid(2, nx=21)
    r = fit(g, eq="u_t = D*u_xx", models=["poly"])
    assert r["poly"].state["deg"] > 0


# ---------------------------------------------------------------- stage F part 2 (black-box reviewer)

def _spring(n=40, noise=0.05, seed=0):
    from scipy.integrate import solve_ivp
    t = np.linspace(0, 10, n)
    s = solve_ivp(lambda tt, y: [y[1], -0.3 * y[1] - 4 * y[0]], (0, 10), [1, 0], t_eval=t, rtol=1e-10).y[0]
    return t, s + noise * np.random.default_rng(seed).standard_normal(n)


@pytest.mark.parametrize("scale", [1e-6, 1e3, 1e6])
def test_g1_ode_fit_works_at_any_time_unit(scale):
    t, u = _spring()
    t0 = time.perf_counter()
    r = fit({"t": t * scale, "u": u}, eq="u_tt + c*u_t + k*u = 0")
    assert time.perf_counter() - t0 < 120
    p = r["equation_fit"].params
    assert abs(p["c"] * scale / 0.3 - 1) < 0.1 and abs(p["k"] * scale ** 2 / 4 - 1) < 0.05, p


def test_g2_tie_rule_is_monotone_and_ignores_one_exploding_prediction():
    from equationnet.fit import Candidate, Result
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal(8)

    def cand(name, e):
        c = Candidate(name, name, name in ("equation_fit", "pinn", "poly_physics"))
        c.fold_errors = [e]
        c.holdout_rmse = float(np.sqrt(np.mean(e ** 2)))
        return c

    boom = noise.copy()
    boom[3] = 1e9                       # one exploding held-out prediction
    r = Result(None, "random", 32, 8, {"poly": cand("poly", noise), "equation_fit": cand("equation_fit", boom)})
    assert r.best.name == "poly" and len(r.tied) == 1
    # monotone: if a worse model is tied, every better one is too
    cs = {"poly": cand("poly", noise), "equation_fit": cand("equation_fit", noise * 1.05),
          "spline": cand("spline", noise * 1.2)}
    r = Result(None, "random", 32, 8, cs)
    order = [c.name for c in r.ranked]
    tied = {c.name for c in r.tied}
    assert all(order[i] in tied for i in range(len(order)) if any(o in tied for o in order[i:]))


@pytest.mark.parametrize("scale", [1e-7, 1e-3, 1, 1e3])
def test_g3_spline_does_not_depend_on_the_time_unit(scale):
    t = np.linspace(0, 5, 30)
    u = 3 * np.exp(-0.7 * t) + 0.02 * np.random.default_rng(23).standard_normal(30)
    r = fit({"t": t * scale, "u": u}, models=["spline"])
    assert r["spline"].ok and abs(r["spline"].holdout_rmse - 0.0281) < 0.002


def test_g5_cli_predict_at_writes_predictions_and_scripts_take_points(tmp_path):
    import os
    from pathlib import Path
    data = csv_file(tmp_path, clean_csv())
    (tmp_path / "fut.csv").write_text("t,u\n6,0\n7,0\n")
    src = str(Path(en.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", BOOT, "fit", EQ, str(data), "--models", "equation_fit",
                          "--predict-at", str(tmp_path / "fut.csv"), "--out", str(tmp_path / "s")],
                         capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    pred = np.loadtxt(tmp_path / "fut_equation_fit_pred.csv", delimiter=",", skiprows=1)
    assert np.allclose(pred[:, 1], 2 * np.exp(-0.7 * pred[:, 0]), atol=0.01)
    assert "ignored" in out.stdout and "u" in out.stdout          # the u column in fut.csv is named
    boot = ("import platform; platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError()); "
            "import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    script = tmp_path / "s" / "equation_fit.py"
    run = subprocess.run([sys.executable, "-I", "-c", boot, str(script), str(data), str(tmp_path / "fut.csv")],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "fut_equation_fit_pred.csv").exists() and "ignored column" in run.stdout
    helped = subprocess.run([sys.executable, "-I", "-c", boot, str(script), "--help"],
                            capture_output=True, text=True, timeout=300)
    assert helped.returncode == 0 and "Usage" in helped.stdout and "Traceback" not in helped.stderr


def test_g6_empty_excel_columns(tmp_path):
    body = clean_csv().splitlines()
    trailing = "\n".join([body[0] + ",,"] + [r + ",," for r in body[1:]]) + "\n"
    r = fit(csv_file(tmp_path, trailing, "a.csv"))                   # empty unnamed columns: dropped
    assert abs(r.best.params["k"] - 0.7) < 0.03
    notes = "\n".join([body[0] + ",notes"] + [r + "," for r in body[1:]]) + "\n"
    stops("'notes'", fit, csv_file(tmp_path, notes, "b.csv"))
    named_less = "\n".join([body[0] + ","] + [r + ",5" for r in body[1:]]) + "\n"
    stops("no name", fit, csv_file(tmp_path, named_less, "c.csv"))


@pytest.mark.parametrize("cv", [1.5, 2.7, True])
def test_g7_cv_must_be_a_whole_number(cv):
    t, u = decay()
    stops("whole number", fit, {"t": t, "u": u}, cv=cv)


def test_g8_ranking_on_very_few_points_is_flagged():
    t, u = decay()
    r = fit({"t": t, "u": u}, holdout=0.03, models=["equation_fit", "poly"])
    assert "rests on only 1 held-out point" in r.table()


def test_g9_settings_for_models_not_run_are_checked():
    t, u = decay()
    stops("does not use", fit, {"t": t, "u": u}, config={"pinn": {"stepz": 10}})


def test_g11_constant_consistent_with_zero_says_so():
    t, u = decay(noise=0.02)
    r = fit({"t": t, "u": u}, eq="u_t = -k*u + c")
    c = r["equation_fit"]
    if abs(c.params["c"]) <= 2 * c.param_se["c"]:
        assert "consistent with 0" in r.table()


# ---- stage A run 1 fixes -------------------------------------------------------------

def test_a5_models_disagreeing_on_a_pinned_constant_is_its_own_note():
    """Stage A run 1: a looser PINN disagreeing with a classical fit that pins a constant
    down was reported as "not pinned down". It is now a separate "models disagree" note;
    with no classical error, or a constant already flagged, it stays a warning."""
    from equationnet.fit import Candidate, Result
    from equationnet.parse import parse_equation
    eq = parse_equation("u_t = -k*u")
    noise = 0.05 * np.random.default_rng(0).standard_normal(20)

    def cand(name, k, se=None):
        c = Candidate(name, name, True)
        c.fold_errors, c.holdout_rmse = [noise], 0.05
        c.params, c.status = {"k": k}, "ok"
        c.param_se = {"k": se} if se is not None else {}
        return c

    r = Result(eq, "random", 80, 20, {"equation_fit": cand("equation_fit", 0.80, 0.004),
                                      "pinn": cand("pinn", 0.90)})
    assert r.constant_notes() == []
    assert len(r.disagreement_notes()) == 1 and "0.8 +- 0.004" in r.disagreement_notes()[0]
    assert "models disagree: k:" in r.table() and "not pinned down" not in r.table()
    # poorly determined: one warning that mentions both
    r = Result(eq, "random", 80, 20, {"equation_fit": cand("equation_fit", 0.80, 0.5),
                                      "pinn": cand("pinn", 1.9)})
    assert len(r.constant_notes()) == 1 and "disagree" in r.constant_notes()[0]
    assert r.disagreement_notes() == []


def test_a_classical_pde_follows_a_pulse_narrower_than_the_data_spacing():
    """Stage A run 1: a Gaussian pulse (width 0.07) on data spaced 0.1 apart gave
    D = -0.001, v = 1.10 (true 0.02, 0.8). Development data (not the stage A seeds)."""
    x, t = np.linspace(0, 2, 21), np.linspace(0, 1, 11)
    XX, TT = np.meshgrid(x, t)
    w = 0.01 + 0.08 * TT
    U = np.sqrt(0.01 / w) * np.exp(-(XX - 0.45 - 0.8 * TT) ** 2 / w)
    U = U + 0.01 * U.std() * np.random.default_rng(77).standard_normal(U.shape)
    r = en.fit("u_t + v*u_x = D*u_xx", {"x": XX.ravel(), "t": TT.ravel(), "u": U.ravel()},
               models=["equation_fit"], split="forecast", verbose=False)
    c = r["equation_fit"]
    for k, true in (("v", 0.8), ("D", 0.02)):
        assert abs(c.params[k] - true) <= 4 * c.param_se[k], (k, c.params[k], c.param_se[k])
    assert abs(c.params["v"] - 0.8) < 0.02 and abs(c.params["D"] - 0.02) < 0.002


def test_a_pde_derivatives_are_fourth_order():
    import equationnet as en2
    ns = {}
    exec(compile(en2.generate("u_t = D*u_xx", "equation_fit", indep=["x", "t"]), "s", "exec"), ns)
    errs = []
    for n in (41, 81):
        x = np.linspace(0, 1, n)
        ux, uxx = ns["_derivatives"](np.sin(3 * x), x[1] - x[0])
        errs.append((np.max(np.abs(ux[2:-2] - 3 * np.cos(3 * x[2:-2]))),
                     np.max(np.abs(uxx[2:-2] + 9 * np.sin(3 * x[2:-2])))))
    assert errs[0][0] / errs[1][0] > 12 and errs[0][1] / errs[1][1] > 12


def _pde_ns(equation):
    ns = {}
    exec(compile(en.generate(equation, "equation_fit", indep=["x", "t"]), "s", "exec"), ns)
    return ns


@pytest.mark.parametrize("case", ["heat_corner_missing", "pulse_oversmoothed"])
def test_a_classical_pde_start_profile_meets_the_edges(case):
    """Stage A run 2: the start profile next to an edge was wrong at t = 0 when the
    earliest data stopped short of the edge (a held-out corner point: the profile was
    held flat to the edge) or when the smoothing flattened a narrow pulse (the profile
    missed the edge value and the correction, pinned to zero there, left a ramp).
    Development data, not the stage A seeds."""
    rng = np.random.default_rng(91)
    x, t = np.linspace(0, 1, 21), np.linspace(0, 1, 11)
    XX, TT = np.meshgrid(x, t)
    if case == "heat_corner_missing":
        eq = "u_t = D*u_xx"
        truth = lambda X: np.sin(np.pi * X[:, 0]) * np.exp(-0.04 * np.pi ** 2 * X[:, 1])
        drop = (TT == 0) & (XX >= 0.9)                  # the start values at x = 0.9, 0.95, 1
    else:
        eq = "u_t + v*u_x = D*u_xx"
        XX, TT = np.meshgrid(2 * x, t)
        truth = lambda X: (np.sqrt(0.01 / (0.01 + 0.08 * X[:, 1]))
                           * np.exp(-(X[:, 0] - 0.55 - 0.7 * X[:, 1]) ** 2 / (0.01 + 0.08 * X[:, 1])))
        drop = (TT == 0) & np.isclose(XX, 0.6)          # the pulse then rests on two points
    X = np.column_stack([XX.ravel(), TT.ravel()])[~drop.ravel()]
    y = truth(X)
    y = y + 0.01 * y.std() * rng.standard_normal(len(y))
    ns = _pde_ns(eq)
    state = ns["train"](X, y)
    lo, hi = X[:, 0].min(), X[:, 0].max()
    Q = np.column_stack([np.concatenate([np.linspace(lo, hi, 81)] * 2), np.repeat([0.0, 0.05], 81)])
    near = (Q[:, 0] - lo <= 0.3) | (hi - Q[:, 0] <= 0.3)     # within 0.3 of an edge
    err = np.where(near, np.abs(ns["predict"](state, Q) - truth(Q)), 0.0)
    assert err.max() < 0.02, (case, float(err.max()), float(Q[np.argmax(err), 0]), float(Q[np.argmax(err), 1]))


def test_a2_a_model_at_the_measurement_noise_is_tied_with_a_lucky_one():
    """Stage A run 2 (Gompertz): a smoothing spline scored well below the noise on 5
    held-out points by luck, the classical fit scored at the noise, and the
    standard-error test called them apart. Both are at the noise floor."""
    from equationnet.fit import Candidate, Result

    def cand(name, e, train):
        c = Candidate(name, name, name in ("equation_fit", "pinn", "poly_physics"))
        c.fold_errors = [np.asarray(e, float)]
        c.holdout_rmse = float(np.sqrt(np.mean(c.fold_errors[0] ** 2)))
        c.train_rmse = train
        return c

    exact = np.array([1.2, -0.9, 1.1, -1.0, 0.8])   # a model right to within noise of size 1
    lucky = 0.4 * exact[::-1]                        # a data-only model, 2.5x lower by luck
    cs = {"spline": cand("spline", lucky, 0.95), "equation_fit": cand("equation_fit", exact, 0.97)}
    r = Result(None, "random", 20, 5, cs)
    assert r.ranked[0].name == "spline" and r.best.name == "equation_fit"
    assert {c.name for c in r.tied} == {"spline", "equation_fit"}
    # no data-only training error, no noise level: the standard-error test alone
    cs["spline"].train_rmse = float("nan")
    assert Result(None, "random", 20, 5, cs).best.name == "spline"
    # a wrong equation, well above the noise, is not tied
    cs = {"spline": cand("spline", lucky, 0.95), "equation_fit": cand("equation_fit", 3 * exact, 0.97)}
    assert Result(None, "random", 20, 5, cs).best.name == "spline"


def test_a_classical_ode_before_the_first_training_point():
    """Stage A run 3 (found by the second preflight recheck): with the earliest points
    held out, the classical fit integrated backwards from its first training point, and
    a fast-relaxing equation multiplied small errors to a held-out error of 78,000.
    The solve now starts where predictions are wanted and runs forwards; a start the
    data cannot see stays near a guess from the first points, and the fit says so. A
    start the data do pin down (an undamped pendulum) is still fitted from the data.
    Development data, not the stage A problems."""
    from scipy.integrate import solve_ivp
    rng = np.random.default_rng(5)
    t = np.linspace(0, 5, 30)
    stiff = solve_ivp(lambda tt, s: [-40 * (s[0] - np.cos(tt))], (0, 5), [1.5], dense_output=True,
                      rtol=1e-11, atol=1e-13, method="Radau")
    pend = solve_ivp(lambda tt, s: [s[1], -12 * np.sin(s[0])], (0, 5), [2.0, 0.0], dense_output=True,
                     rtol=1e-11, atol=1e-13)
    for eq, sol, scale in (("u_t = -k*(u - cos(t))", stiff, 1.0), ("u_tt = -w*sin(u)", pend, 1.0)):
        u = sol.sol(t)[0]
        y = u + 0.01 * u.std() * rng.standard_normal(len(t))
        keep = t > 0.2                                  # the first two points are not trained on
        ns = {}
        exec(compile(en.generate(eq, "equation_fit"), "s", "exec"), ns)
        state = ns["train"](t[keep][:, None], y[keep], domain=[[0.0], [5.0]])
        Q = np.linspace(0, 5, 201)
        err = np.abs(ns["predict"](state, Q[:, None]) - sol.sol(Q)[0])
        if eq.startswith("u_t ="):
            # u(0) = 1.5 is unknowable here; the guess is the curve through the first points
            assert state["t0"] == 0.0 and "not determined by the data" in (state.get("note") or "")
            assert err.max() < 1.0 and err[Q > 0.25].max() < 0.02, (err.max(), err[Q > 0.25].max())
        else:
            assert "not determined" not in (state.get("note") or "")
            assert err.max() < 0.05, err.max()


def test_a6_profile_flags_constants_on_a_curved_valley():
    """Stage A run 4: a second-order equation fitted to a single exponential has a whole
    valley of equally good (c, k), but where the fit landed k = -0.55 +- 0.24 looked
    pinned down, so only c was flagged. The profile check moves each constant by half
    and refits the rest. Development data, not the stage A problems."""
    t = np.linspace(0, 6, 25)
    # noise seed 10: without the profile check c = 1.12 +- 0.48 looks pinned down
    u = 3 * np.exp(-0.7 * t) + 0.01 * 3 * np.random.default_rng(10).standard_normal(25)
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["equation_fit"], verbose=False)
    notes = " ".join(r.constant_notes())
    assert notes.count("c:") == 1 and notes.count("k:") == 1, r.constant_notes()


def test_a_moved_start_keeps_the_constants_standard_errors():
    """Third preflight recheck: when the start was moved before the first training
    point, the weak pull toward the start guess was counted as data, which made every
    standard error too small by sqrt((m - p) / (m - p + ORDER)) (Misra1a: exactly
    sqrt(8/9)). Where the start sits must not change a constant's standard error.
    Development data, not the stage A or G sets."""
    t = np.linspace(0, 4, 21)
    u = 5 * np.exp(-0.8 * t) + 0.02 * np.random.default_rng(12).standard_normal(21)
    ns = {}
    exec(compile(en.generate("u_t = -k*u", "equation_fit"), "s", "exec"), ns)
    keep = t > 0.1                                       # the first point is not trained on
    plain = ns["train"](t[keep][:, None], u[keep])
    moved = ns["train"](t[keep][:, None], u[keep], domain=[[0.0], [4.0]])
    assert moved["t0"] == 0.0 and plain["t0"] > 0
    ratio = moved["se"]["k"] / plain["se"]["k"]
    assert abs(ratio - 1) < 0.005, ratio


def test_g_classical_pde_says_when_it_predicts_past_the_trained_times():
    # stage G (soil): on a forecast the edge values after the last training time are
    # not known and are held; the table must say so. Inside the trained times, silence.
    x, t = np.linspace(0, 1, 11), np.linspace(0, 0.5, 8)
    XX, TT = np.meshgrid(x, t)
    U = np.exp(-np.pi ** 2 * 0.1 * TT) * np.sin(np.pi * XX) + 0.3 * XX
    data = {"x": XX.ravel(), "t": TT.ravel(), "u": U.ravel()}
    fore = en.fit("u_t = a*u_xx", data, models=["equation_fit"], split="forecast", verbose=False)
    note = fore["equation_fit"].note
    assert note and "after t = " in note and "held at their last fitted values" in note
    assert "note (Equation fit (classical)): Outside the trained times" in fore.table()
    rand = en.fit("u_t = a*u_xx", data, models=["equation_fit"], split="random", verbose=False)
    assert rand["equation_fit"].note is None
    early = en.fit("u_t = a*u_xx", data, models=["equation_fit"], split="random",
                   predict_at={"x": [0.5], "t": [-0.1]}, verbose=False)
    assert "before t = " in early["equation_fit"].note
