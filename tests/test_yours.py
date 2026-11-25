"""Your own models (en.custom / YourModel): scored on the same held-out points as the
built-in ones, never preferred on a tie, and kept out of the checks that assume a model's
constants are the equation's."""

import numpy as np
import pytest

import equationnet as en
from equationnet.fit import _noise_floor

EQ = "u_t = a + b*t"     # any equation in t works: these tests run data-only built-ins


def line(n=60, noise=0.05, seed=0):
    t = np.linspace(0, 5, n)
    return {"t": t, "u": 1 + 2 * t + noise * np.random.default_rng(seed).standard_normal(n)}


class Mean:
    """Predicts the training mean everywhere: a bad model."""
    def fit(self, X, y):
        self.m = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class Line:
    """Least-squares straight line, with constants and standard errors."""
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X[:, 0]])
        self.p, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        s2 = float(np.sum((A @ self.p - y) ** 2) / (len(y) - 2))
        self.se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
        return self

    def predict(self, X):
        return self.p[0] + self.p[1] * X[:, 0]

    def fitted_params(self):
        return {"a": self.p[0], "b": self.p[1]}

    def uncertainty(self):
        return {"a": self.se[0], "b": self.se[1]}


class Interpolator:
    """Passes through every training point (training error 0)."""
    def fit(self, X, y):
        o = np.argsort(X[:, 0])
        self.x, self.y = X[o, 0], y[o]
        return self

    def predict(self, X):
        return np.interp(X[:, 0], self.x, self.y)


def truth(X):
    return 1 + 2 * X[:, 0]


def run(models, **kw):
    return en.fit(EQ, line(), models=models, verbose=False, **kw)


def test_scored_on_the_same_points_as_the_built_ins():
    r = run(["poly", Mean(), Line()])
    assert {"poly", "Mean", "Line"} == set(r.candidates)
    for c in r.candidates.values():
        assert c.ok, c.status
        assert len(c.holdout_errors) == r.n_holdout
    # the same held-out points: a straight line and the polynomial see the same residual signs
    assert r["Line"].yours and not r["poly"].yours
    assert r["Mean"].holdout_rmse > 10 * r["Line"].holdout_rmse
    assert r["Line"].title == "Line (yours)"


def test_adding_yours_does_not_change_the_built_in_scores():
    a = run(["poly", "spline"])
    b = run(["poly", "spline", Mean(), Interpolator()])
    for n in ("poly", "spline"):
        assert a[n].holdout_rmse == b[n].holdout_rmse


def test_constants_and_standard_errors_are_shown():
    r = run([Line()])
    assert set(r["Line"].params) == {"a", "b"}
    assert abs(r["Line"].params["b"] - 2) < 0.05
    assert r["Line"].param_se["b"] > 0
    assert "b=2" in r.table() and "+-" in r.table()


def test_plain_function_is_used_as_it_is():
    r = run([truth, lambda X: 0 * X[:, 0]])
    assert r["truth"].ok and r["your_model_2"].ok
    assert r["truth"].holdout_rmse < 0.1 < r["your_model_2"].holdout_rmse
    assert "did not train it" in r["truth"].script


def test_your_object_is_not_changed():
    m = Line()
    r = run([m])
    assert not hasattr(m, "p")                   # a copy was trained
    assert hasattr(r["Line"].state, "p")
    assert r["Line"].predict([[1.0]]).shape == (1,)


def test_cross_validation_trains_per_fold_and_on_everything():
    r = run([Line()], cv=4)
    assert len(r["Line"].fold_rmse) == 4
    assert r.n_train == 60


def test_tie_never_resolves_to_yours():
    r = run(["poly", truth])
    names = {c.name for c in r.tied}
    assert names == {"poly", "truth"}            # both at the noise
    assert r.best.name == "poly"
    if r.ranked[0].name == "truth":
        assert "Yours has the lowest error, but on a tie a built-in model is" in r.table()
    assert run(["poly", truth], pick="lowest").best.name == r.ranked[0].name


def test_yours_wins_when_clearly_better():
    r = run(["spline", Mean(), truth], holdout=0.3)
    assert r.best.name in ("truth", "spline")
    r = en.fit(EQ, line(), models=[Mean(), truth], verbose=False)
    assert r.best.name == "truth"                # nothing built in to prefer


def test_noise_floor_ignores_yours():
    r = run(["poly", "spline", Interpolator()])
    built_in = [c for c in r.candidates.values() if not c.yours]
    n = r.n_holdout
    assert _noise_floor(r.candidates.values(), n) == _noise_floor(built_in, n)
    assert r["Interpolator"].train_rmse == 0.0


def test_your_constants_stay_out_of_the_equation_checks():
    class HalfB(Line):                            # 'b' here is half the equation's b
        def fitted_params(self):
            return {"a": self.p[0], "b": self.p[1] / 2}
    r = run(["poly", en.custom(HalfB(), name="half", uses_equation=True), Line()])
    assert r["half"].uses_equation
    assert r["half"].title == "half (yours, uses equation)"
    assert r.disagreement_notes() == [] and r.constant_notes() == []


def test_failures_are_reported_not_raised():
    class Broken(Line):
        def fit(self, X, y):
            raise RuntimeError("boom")

    class WrongLength(Line):
        def predict(self, X):
            return np.zeros(len(X) + 1)

    class Nan(Line):
        def predict(self, X):
            return np.full(len(X), np.nan)

    r = run(["poly", Broken(), WrongLength(), Nan()])
    assert r["poly"].ok
    assert r["Broken"].status == "failed: RuntimeError: boom"
    assert "must return" in r["WrongLength"].status
    assert "NaN" in r["Nan"].status
    assert r.best.name == "poly"


def test_column_predictions_are_accepted():
    class Column(Line):
        def predict(self, X):
            return super().predict(X).reshape(-1, 1)
    assert run([Column()])["Column"].ok


@pytest.mark.parametrize("bad, words", [
    (Line, "pass an instance"),
    (42, "is not a model"),
    (en.custom(Line(), name="poly"), "'poly' is the name of a built-in model"),
    (en.custom(Line(), name="has space"), "cannot be used"),
])
def test_bad_models_are_named(bad, words):
    with pytest.raises(ValueError, match=words):
        run(["poly", bad])


def test_two_of_yours_with_one_name():
    with pytest.raises(ValueError, match="two models are called 'Line'"):
        run([Line(), Line()])
    r = run([Line(), en.custom(Line(), name="line2")])
    assert {"Line", "line2"} == set(r.candidates)


def test_stub_is_saved(tmp_path):
    r = run(["poly", Line()])
    paths = r.save_scripts(tmp_path)
    stub = (tmp_path / "Line.py").read_text(encoding="utf-8")
    assert tmp_path / "Line.py" in paths
    assert "This is a stub" in stub and "model.fit(X_train, y_train)" in stub
    compile(stub, "Line.py", "exec")             # valid Python, does nothing
    assert "Saved scripts for yours are stubs" in r.table()


def test_single_model_and_strings_still_work():
    assert set(run("poly").candidates) == {"poly"}
    assert set(run(Line()).candidates) == {"Line"}
    assert set(run(("poly", "poly", Line())).candidates) == {"poly", "Line"}


def test_name_error_says_what_is_wrong():
    with pytest.raises(ValueError, match="it contains ' '"):
        run([en.custom(Line(), name="my GP")])


def test_stub_records_public_import_and_parameters():
    class Params(Line):
        def __init__(self, alpha=0.5):
            self.alpha = alpha

        def get_params(self, deep=True):
            return {"alpha": self.alpha}
    Params.__module__, Params.__qualname__ = "somelib.models._private", "Params"
    stub = run([Params(alpha=2.0)])["Params"].script
    assert "# from somelib.models import Params" in stub
    assert "alpha=2.0" in stub


def test_underfitting_built_ins_do_not_tie_with_a_much_better_model_of_yours():
    """Preflight V1: the built-in data-only models all underfit, so their training
    error (the usual noise level) is their misfit; they must not tie with yours."""
    t = np.linspace(0, 5, 120)
    u = 1 + 2 * t + 0.5 * np.sin(25 * t) + np.random.default_rng(1).normal(0, 0.01, 120)

    def exact(X):
        return 1 + 2 * X[:, 0] + 0.5 * np.sin(25 * X[:, 0])
    r = en.fit("u_t = a + b*t", {"t": t, "u": u}, models=["poly", "spline", exact], verbose=False)
    assert r.ranked[0].name == "exact"
    assert [c.name for c in r.tied] == ["exact"]
    assert r.best.name == "exact"
    assert "within the noise" not in r.table()


def test_tie_between_two_of_yours_says_so():
    def a(X):
        return 1 + 2 * X[:, 0]

    def b(X):
        return 1 + 2 * X[:, 0] + 1e-4
    r = run([a, en.custom(b, uses_equation=True)])
    if len(r.tied) == 2 and r.ranked[0].name == "a":
        assert r.best.name == "b"
        assert "Of your tied models, one you marked as using the equation is returned" in r.table()
    assert "on a tie a built-in model is returned" not in r.table()


def test_your_standard_errors_get_their_own_legend():
    t = run([Line()]).table()
    assert "+- on your models: as your model's uncertainty() reports it." in t
    assert "if the equation is exactly right" not in t


def test_a_much_worse_model_of_yours_is_not_tied_with_a_built_in_lowest():
    """Preflight recheck R2: the built-in data-only models underfit (inflated noise
    level); the classical fit of the right equation is lowest; a model of yours at
    3 times its error must not be marked tied with it."""
    t = np.linspace(0, 5, 120)
    u = 1 + 2 * t + 0.5 * np.sin(25 * t) + np.random.default_rng(1).normal(0, 0.01, 120)

    def worse(X):
        return 1 + 2 * X[:, 0] + 0.5 * np.sin(25 * X[:, 0]) + 0.02
    r = en.fit("u_t = 2 + 12.5*b*cos(25*t)", {"t": t, "u": u},
               models=["poly", "spline", "equation_fit", worse], verbose=False)
    assert r.ranked[0].name == "equation_fit", r.table()
    assert r["worse"].holdout_rmse > 2 * r.ranked[0].holdout_rmse
    assert "worse" not in [c.name for c in r.tied]


class Cubic:
    def fit(self, X, y):
        self.c = np.polyfit(X[:, 0], y, 3)
        return self

    def predict(self, X):
        return np.polyval(self.c, X[:, 0])


def test_a_lucky_model_of_yours_does_not_untie_built_ins_at_the_noise():
    """Second recheck N2: 5 held-out points; a data-only cubic of yours scores well
    below the noise by luck. The built-in models at the noise must stay tied with it,
    so the classical fit is returned, as without the cap."""
    t = np.linspace(0, 5, 25)
    u = 2 * np.exp(-0.7 * t) + np.random.default_rng(23).normal(0, 0.05, 25)
    r = en.fit("u_t = -k*u", {"t": t, "u": u}, models=["poly", "spline", "equation_fit", Cubic()],
               seed=23, verbose=False)
    if r.ranked[0].name == "Cubic":
        assert r.best.name == "equation_fit", r.table()


def test_marks_out_of_error_order_are_explained():
    t = np.linspace(0, 5, 120)
    u = 1 + 2 * t + 0.5 * np.sin(25 * t) + np.random.default_rng(1).normal(0, 0.01, 120)

    def worse(X):
        return 1 + 2 * X[:, 0] + 0.5 * np.sin(25 * X[:, 0]) + 0.02
    r = en.fit("u_t = 2 + 12.5*b*cos(25*t)", {"t": t, "u": u},
               models=["poly", "spline", "equation_fit", worse], verbose=False)
    if any(c.name in ("poly", "spline") for c in r.tied):
        assert "worse (yours): not marked tied, though built-in models with larger errors are" in r.table()


def test_on_five_held_out_points_a_much_worse_model_of_yours_is_not_tied():
    """Third recheck M1: on small held-out sets a model of yours at 2-4 times the lowest
    (a built-in) was marked tied; yours now tie only within the strict cap."""
    t = np.linspace(0, 5, 25)
    rng = np.random.default_rng(0)
    for seed in range(6):
        u = 1 + 2 * t + 0.5 * np.sin(25 * t) + np.random.default_rng(seed).normal(0, 0.01, 25)

        def worse(X):
            return 1 + 2 * X[:, 0] + 0.5 * np.sin(25 * X[:, 0]) + 0.03
        r = en.fit("u_t = 2 + 12.5*b*cos(25*t)", {"t": t, "u": u}, seed=seed,
                   models=["poly", "spline", "equation_fit", worse], verbose=False)
        assert r.n_holdout == 5
        low = r.ranked[0]
        if not low.yours and r["worse"].holdout_rmse > 2 * low.holdout_rmse:
            assert "worse" not in [c.name for c in r.tied], (seed, r.table())
