import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import equationnet as en
from equationnet.cli import main

FAST = {"pinn": {"steps": 150, "lbfgs": 20, "collocation": 128},
        "poly_physics": {"steps": 150, "lbfgs": 20, "collocation": 128},
        "nn": {"steps": 150, "lbfgs": 20}}


def oscillator(n=40, noise=0.0, seed=0):
    t = np.linspace(0, 6, n)
    sol = solve_ivp(lambda tt, s: [s[1], -0.3 * s[1] - 4.0 * s[0]], (0, 6), [1.0, 0.0],
                    t_eval=t, rtol=1e-10, atol=1e-12)
    return t, sol.y[0] + noise * np.random.default_rng(seed).standard_normal(n)


def heat(nx=11, nt=11, D=0.1):
    x, t = np.linspace(0, 1, nx), np.linspace(0, 0.5, nt)
    XX, TT = np.meshgrid(x, t)
    U = np.exp(-D * np.pi ** 2 * TT) * np.sin(np.pi * XX)
    return {"x": XX.ravel(), "t": TT.ravel(), "u": U.ravel()}


def test_every_model_runs_on_an_ode():
    t, u = oscillator()
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, config=FAST, verbose=False)
    for c in r.candidates.values():
        assert c.ok, (c.name, c.status)
        assert np.isfinite(c.holdout_rmse)
    assert set(r["pinn"].params) == {"c", "k"}
    assert r["nn"].params == {}
    assert "best:" in r.table()


def test_classical_fit_recovers_constants_on_clean_data():
    t, u = oscillator()
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["equation_fit"], verbose=False)
    p = r["equation_fit"].params
    assert abs(p["c"] - 0.3) < 1e-3 and abs(p["k"] - 4.0) < 1e-3


def test_classical_ode_finds_the_frequency_and_flags_the_alias():
    """20 evenly spaced noisy points over ~3 swings: the regression start is useless here and
    a spring at k~99 passes through the same sample points. The fit must find k=4, pick it
    over the alias, and say an alias exists."""
    t = np.linspace(0, 10, 20)
    sol = solve_ivp(lambda tt, s: [s[1], -0.3 * s[1] - 4.0 * s[0]], (0, 10), [1.0, 0.0],
                    t_eval=t, rtol=1e-10, atol=1e-12)
    u = sol.y[0] + 0.05 * np.random.default_rng(0).standard_normal(len(t))
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["equation_fit"],
               holdout=0.05, verbose=False)
    c = r["equation_fit"]
    assert abs(c.params["k"] - 4.0) < 0.2 and abs(c.params["c"] - 0.3) < 0.1
    assert c.note and "equally well" in c.note and "note (" in r.table()


@pytest.mark.parametrize("model", ["equation_fit", "poly_physics", "pinn"])
def test_constants_far_from_one_are_found(model):
    """Real yeast data has a carrying capacity near 665. The physics models used to start
    every constant at 1 and stall there (K = 1.6)."""
    t = np.arange(19.0)
    u = 665 / (1 + (665 / 9.6 - 1) * np.exp(-0.53 * t))
    r = en.fit("u_t = r*u*(1 - u/K)", {"t": t, "u": u}, models=[model], config=FAST, verbose=False)
    p = r[model].params
    assert abs(p["K"] / 665 - 1) < 0.05 and abs(p["r"] / 0.53 - 1) < 0.1, p


def test_standard_errors_cover_the_truth_and_print():
    t = np.arange(19.0)
    clean = 665 / (1 + (665 / 9.6 - 1) * np.exp(-0.53 * t))
    u = clean + 5.0 * np.random.default_rng(125).standard_normal(19)   # starts near zero: a noisy
    r = en.fit("u_t = r*u*(1 - u/K)", {"t": t, "u": u},                 # smooth curve dips below
               models=["equation_fit"], holdout=0.05, verbose=False)     # it, which used to break the fit
    c = r["equation_fit"]
    for name, true in (("K", 665), ("r", 0.53)):
        assert 0 < c.param_se[name] < 0.05 * true
        assert abs(c.params[name] - true) < 3 * c.param_se[name]
    assert "K=" in r.table() and "+-" in r.table() and "not pinned down" not in r.table()


def test_a_constant_the_data_cannot_determine_is_flagged():
    """u_t = -(a + b)*u: only a + b is determined, so a and b each get an error far larger
    than their values (infinite in exact arithmetic; the fit's numerical gradients leave
    a trace of difference between the two)."""
    t = np.linspace(0, 3, 30)
    u = 2 * np.exp(-0.8 * t) + 0.01 * np.random.default_rng(0).standard_normal(30)
    r = en.fit("u_t = -(a + b)*u", {"t": t, "u": u}, models=["equation_fit"], verbose=False)
    c = r["equation_fit"]
    assert abs(c.params["a"] + c.params["b"] - 0.8) < 0.02
    for name in ("a", "b"):
        assert c.param_se[name] > 2 * abs(c.params[name])
    table = r.table()
    assert "not pinned down: a:" in table and "known={'a': ...}" in table


def test_classical_pde_reports_a_standard_error():
    r = en.fit("u_t = D*u_xx", heat(), models=["equation_fit"], verbose=False)
    se = r["equation_fit"].param_se["D"]
    assert 0 < se < 0.01


def test_classical_pde_recovers_diffusivity():
    r = en.fit("u_t = D*u_xx", heat(), models=["equation_fit"], verbose=False)
    assert abs(r["equation_fit"].params["D"] - 0.1) < 0.01


def test_not_applicable_is_stated_in_words():
    d = heat()
    r = en.fit("u_tt = c^2*u_xx", d, models=["equation_fit"], verbose=False)
    st = r["equation_fit"].status
    assert st.startswith("not applicable") and "u_tt" in st


def test_forecast_split_holds_out_the_end():
    t, u = oscillator()
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["spline"],
               split="forecast", cv=1, verbose=False)
    assert r.folds == 1 and r.n_holdout == 8 and r.n_train == 32


def test_small_data_is_cross_validated_and_retrained_on_everything():
    t, u = oscillator(40, noise=0.02)
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["spline", "poly"], cv="auto",
               verbose=False)
    assert r.folds == 5 and r.n_train == 40 and r.n_holdout == 8
    c = r["spline"]
    assert len(c.fold_rmse) == 5 and c.holdout_rmse == pytest.approx(np.mean(c.fold_rmse))
    assert "5-fold cross-validation" in r.table() and "retrained on all 40" in r.table()
    # large data keeps one split
    t2, u2 = oscillator(60)
    r2 = en.fit("u_tt + c*u_t + k*u = 0", {"t": t2, "u": u2}, models=["spline"], cv="auto",
                verbose=False)
    assert r2.folds == 1 and r2.n_train == 48
    # the default is one split
    r3 = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["spline"], verbose=False)
    assert r3.folds == 1 and r3.n_train == 32


def test_forecast_folds_always_train_on_the_past():
    from equationnet.fit import _folds
    t = np.linspace(0, 1, 40)
    folds = _folds(t[:, None], t, ["t"], "forecast", 0.2, 0, 3)
    assert len(folds) == 3
    for train, test in folds:
        assert test.sum() == 8 and train.sum() >= 16      # windows do not overlap (stage F)
        assert t[train].max() < t[test].min()
    assert folds[0][1][-1] and not folds[1][1][-1]      # first origin = the ordinary split


def test_random_folds_score_every_point_once():
    from equationnet.fit import _folds
    t = np.linspace(0, 1, 23)
    folds = _folds(t[:, None], t, ["t"], "random", 0.2, 3, 5)
    assert np.array_equal(sum(te.astype(int) for _, te in folds), np.ones(23, int))


def test_predict_at_extends_where_the_equation_is_enforced(monkeypatch):
    F = sys.modules["equationnet.fit"]
    seen = []
    real = F._train_eval
    monkeypatch.setattr(F, "_train_eval", lambda *a: seen.append(a[-1]) or real(*a))
    t, u = oscillator(40)
    en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["pinn"], cv=1,
           predict_at={"t": [9.0]}, config=FAST, verbose=False)
    assert seen and seen[0][1][0] == 9.0 and seen[0][0][0] == 0.0


def test_classical_pde_fills_in_a_missing_start_profile():
    """With the first time slice and most of the left edge removed, the start profile and
    edges are fitted instead of read off the data."""
    d = heat()
    keep = ~((d["t"] == 0) & (d["x"] > 0.15) & (d["x"] < 0.85))   # leaves 3 points at t=0
    keep &= ~((d["x"] == 0) & (d["t"] > 0.1))
    d = {k: v[keep] for k, v in d.items()}
    r = en.fit("u_t = D*u_xx", d, models=["equation_fit"], cv=1, holdout=0.05, verbose=False)
    c = r["equation_fit"]
    assert c.ok, c.status
    assert abs(c.params["D"] - 0.1) < 0.01
    x = np.linspace(0, 1, 21)
    start = c.predict(np.column_stack([x, np.zeros_like(x)]))
    assert np.max(np.abs(start - np.sin(np.pi * x))) < 0.05


def test_known_constant_is_not_fitted():
    t, u = oscillator()
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, known={"k": 4.0},
               models=["equation_fit"], verbose=False)
    assert set(r["equation_fit"].params) == {"c"}


def test_generate_without_data_mentions_equation():
    s = en.generate("u_t = D*u_xx", "pinn")
    assert "u_t = D*u_xx" in s and "def residual" in s and "def train" in s


def test_settings_are_never_ignored_silently():
    eq = "u_tt + c*u_t + k*u = 0"
    a = en.generate(eq, "pinn", config={"steps": 123})
    assert a == en.generate(eq, "pinn", config={"pinn": {"steps": 123}}) and "STEPS = 123" in a
    with pytest.raises(ValueError, match="does not use"):
        en.generate(eq, "spline", config={"steps": 123})
    with pytest.raises(ValueError, match="unknown model"):
        en.fit(eq, {"t": [0, 1, 2, 3], "u": [1, 0, -1, 0]}, config={"pnn": {"steps": 1}}, verbose=False)


def test_tie_rule_returns_the_simplest_model_within_noise():
    from equationnet.fit import Candidate, Result
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal(30)

    def cand(name, errors):
        c = Candidate(name, name, name in ("equation_fit", "pinn"))
        c.fold_errors = [errors]
        c.holdout_rmse = float(np.sqrt(np.mean(errors ** 2)))
        return c

    # the classical fit makes the same size of errors, on different points, 1% larger
    tie = {"pinn": cand("pinn", noise), "equation_fit": cand("equation_fit", 1.01 * noise[::-1]),
           "nn": cand("nn", noise + 0.2)}
    r = Result(None, "random", 100, 30, tie)
    assert r.ranked[0].name == "pinn"
    assert {c.name for c in r.tied} == {"pinn", "equation_fit"} and r.best.name == "equation_fit"
    assert Result(None, "random", 100, 30, tie, pick="lowest").best.name == "pinn"
    clear = {"pinn": cand("pinn", noise), "equation_fit": cand("equation_fit", noise + 0.1)}
    r2 = Result(None, "random", 100, 30, clear)
    assert [c.name for c in r2.tied] == ["pinn"] and r2.best.name == "pinn"


def test_tie_is_explained_in_the_table():
    t = np.linspace(0, 10, 40)
    sol = solve_ivp(lambda tt, s: [s[1], -0.3 * s[1] - 4.0 * s[0]], (0, 10), [1.0, 0.0],
                    t_eval=t, rtol=1e-10, atol=1e-12)
    u = sol.y[0] + 0.05 * np.random.default_rng(1).standard_normal(40)
    r = en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["equation_fit", "poly", "spline"],
               verbose=False)
    if len(r.tied) > 1:
        assert "within the noise" in r.table() and r.best.name == "equation_fit"
    with pytest.raises(ValueError, match="pick"):
        en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["poly"], pick="best", verbose=False)


def test_models_run_fastest_first(capsys):
    t, u = oscillator()
    en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["equation_fit", "poly", "spline"])
    out = capsys.readouterr().out
    assert out.index("Polynomial") < out.index("Smoothing spline") < out.index("Equation fit")


@pytest.mark.parametrize("model", ["equation_fit", "pinn", "spline", "poly"])
def test_exported_script_runs_standalone(tmp_path, model):
    t, u = oscillator(30)
    csv = tmp_path / "data.csv"
    np.savetxt(csv, np.column_stack([t, u]), delimiter=",", header="t,u", comments="")
    script = en.generate("u_tt + c*u_t + k*u = 0", model,
                         config={"pinn": {"steps": 100, "lbfgs": 10, "collocation": 64}}
                         if model == "pinn" else None)
    path = tmp_path / f"{model}.py"
    path.write_text(script, encoding="utf-8")
    boot = ("import platform; platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError()); "
            "import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    out = subprocess.run([sys.executable, "-c", boot, str(path), str(csv)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "rms error on the data" in out.stdout
    assert (tmp_path / f"data_{model}_pred.csv").exists()


def test_cli_script_and_models(capsys):
    assert main(["script", "u_t = -k*u", "--model", "equation_fit"]) == 0
    assert "def highest" in capsys.readouterr().out
    assert main(["models"]) == 0
    assert main(["script", "u_t = 1 = 2"]) == 2


def test_fit_leaves_the_callers_torch_settings_alone():
    """Torch models train on one thread in float64, then restore the caller's settings."""
    import torch
    torch.set_num_threads(3)
    before = torch.get_default_dtype(), torch.get_num_threads()
    t, u = oscillator()
    en.fit("u_tt + c*u_t + k*u = 0", {"t": t, "u": u}, models=["pinn", "nn", "poly_physics"],
           config=FAST, verbose=False)
    assert (torch.get_default_dtype(), torch.get_num_threads()) == before
