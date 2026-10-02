"""Tests for Bayesian Optimization wrappers (BayesOptClassifier, BayesOptRegressor)."""

import numpy as np
import pytest

from audio_benchmark.bayes_opt import (
    BayesOptClassifier,
    BayesOptRegressor,
    _sample_param,
    get_default_classifier_param_space,
    get_default_regressor_param_space,
)
from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers import KNNBenchmarkClassifier, create_classifier
from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.regressors import LinearBenchmarkRegressor, create_regressor
from audio_benchmark.tasks.global_classification import GlobalClassificationTask
from audio_benchmark.tasks.regression import RegressionTask


@pytest.fixture
def synthetic_classification_data():
    np.random.seed(42)
    # 24 samples, 4 features, 2 balanced classes
    X = np.random.randn(24, 4).astype(np.float32)
    y = np.array([0] * 12 + [1] * 12)
    X[y == 1] += 2.0
    return X[:18], y[:18], X[18:], y[18:]


@pytest.fixture
def synthetic_regression_data():
    np.random.seed(42)
    # 24 samples, 4 features, continuous linear target with noise
    X = np.random.randn(24, 4).astype(np.float32)
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.2 * np.random.randn(24).astype(np.float32)
    return X[:18], y[:18], X[18:], y[18:]


def test_sample_param_discrete_and_continuous():
    class DummyTrial:
        def __init__(self):
            self.calls = []

        def suggest_categorical(self, name, choices):
            self.calls.append(("categorical", name, choices))
            return choices[0]

        def suggest_int(self, name, low, high, step=1):
            self.calls.append(("int", name, low, high, step))
            return low

        def suggest_float(self, name, low, high, log=False):
            self.calls.append(("float", name, low, high, log))
            return low

    trial = DummyTrial()

    # Discrete categorical list
    val = _sample_param(trial, "metric", ["minkowski", "cosine"])
    assert val == "minkowski"

    # Discrete int tuple
    val = _sample_param(trial, "n_neighbors", ("int", 1, 10))
    assert val == 1

    # Discrete int tuple without string tag
    val = _sample_param(trial, "n_neighbors", (1, 10))
    assert val == 1

    # Continuous float
    val = _sample_param(trial, "lr", ("float", 1e-4, 1e-1, True))
    assert val == 1e-4

    # Float with "log" string
    val = _sample_param(trial, "alpha", (1e-3, 10.0, "log"))
    assert val == 1e-3

    # Dict specification for categorical and int
    val_cat = _sample_param(
        trial, "agg", {"type": "categorical", "choices": ["cls", "mean"]}
    )
    assert val_cat == "cls"

    val_int = _sample_param(trial, "dim", {"type": "int", "low": 16, "high": 64})
    assert val_int == 16


def test_default_search_spaces():
    for name in ("linear", "knn", "mlp", "transformer"):
        space_clf = get_default_classifier_param_space(name)
        assert isinstance(space_clf, dict)
        assert len(space_clf) > 0

        space_reg = get_default_regressor_param_space(name)
        assert isinstance(space_reg, dict)
        assert len(space_reg) > 0


def test_bayes_opt_classifier_knn(synthetic_classification_data):
    X_train, y_train, X_test, y_test = synthetic_classification_data

    clf = BayesOptClassifier(
        base_model="knn",
        n_trials=5,
        cv=2,
        random_state=42,
        show_progress=False,
    )
    clf.fit(X_train, y_train)

    assert clf.best_model is not None
    assert isinstance(clf.best_model, KNNBenchmarkClassifier)
    assert isinstance(clf.best_params, dict)
    assert "n_neighbors" in clf.best_params
    assert clf.best_score is not None

    preds = clf.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = clf.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_bayes_opt_classifier_custom_param_space(synthetic_classification_data):
    X_train, y_train, X_test, y_test = synthetic_classification_data

    custom_space = {
        "n_neighbors": [1, 3],
        "metric": ["minkowski"],
    }
    clf = BayesOptClassifier(
        base_model="knn",
        param_space=custom_space,
        n_trials=4,
        cv=2,
        random_state=42,
        show_progress=False,
    )
    clf.fit(X_train, y_train)

    assert clf.best_params["n_neighbors"] in [1, 3]
    assert clf.best_params["metric"] == "minkowski"
    metrics = clf.evaluate(X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_bayes_opt_classifier_callable_space(synthetic_classification_data):
    X_train, y_train, X_test, y_test = synthetic_classification_data

    def custom_sampler(trial):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 3),
            "metric": "minkowski",
        }

    clf = BayesOptClassifier(
        base_model="knn",
        param_space=custom_sampler,
        n_trials=3,
        cv=2,
        random_state=42,
        show_progress=False,
    )
    clf.fit(X_train, y_train)
    assert 1 <= clf.best_params["n_neighbors"] <= 3
    metrics = clf.evaluate(X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_bayes_opt_classifier_via_factory():
    clf = create_classifier(
        "bayes_opt",
        base_model="knn",
        n_trials=3,
        cv=2,
        show_progress=False,
    )
    assert isinstance(clf, BayesOptClassifier)
    assert clf.model_name == "knn"


def test_bayes_opt_regressor_linear(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data

    reg = BayesOptRegressor(
        base_model="linear",
        n_trials=5,
        cv=2,
        metric="r2",
        random_state=42,
        show_progress=False,
    )
    reg.fit(X_train, y_train)

    assert reg.best_model is not None
    assert isinstance(reg.best_model, LinearBenchmarkRegressor)
    assert "alpha" in reg.best_params
    assert reg.best_score is not None

    preds = reg.predict(X_test)
    assert len(preds) == len(X_test)

    metrics = reg.evaluate(X_test, y_test)
    assert "mse" in metrics
    assert "r2" in metrics
    assert "mae" in metrics


def test_bayes_opt_regressor_minimize_metric(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data

    reg = BayesOptRegressor(
        base_model="linear",
        n_trials=4,
        cv=2,
        metric="mse",
        random_state=42,
        show_progress=False,
    )
    reg.fit(X_train, y_train)
    assert reg.study.direction.name == "MINIMIZE"
    metrics = reg.evaluate(X_test, y_test)
    assert "mse" in metrics


def test_bayes_opt_regressor_via_factory():
    reg = create_regressor(
        "bayes_opt",
        base_model="knn",
        n_trials=3,
        cv=2,
        show_progress=False,
    )
    assert isinstance(reg, BayesOptRegressor)
    assert reg.model_name == "knn"


def test_bayes_opt_in_task_and_benchmark(tmp_path):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=False)
    dataset = DummyAudioDataset(num_samples=16, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=8)

    opt_classifier = BayesOptClassifier(
        base_model="knn",
        param_space={"n_neighbors": [1, 3]},
        n_trials=3,
        cv=2,
        show_progress=False,
    )
    task = GlobalClassificationTask(classifier=opt_classifier)

    result = benchmark.run(
        model=model,
        model_name="dummy_opt_encoder",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert "accuracy" in result.metrics
    assert "best_params" in result.metrics
    assert result.metrics["best_params"]["n_neighbors"] in [1, 3]
    assert "best_cv_score" in result.metrics

    # Check print_summary prints cleanly
    result.print_summary()


def test_bayes_opt_regression_task_and_benchmark(tmp_path):
    class DummyRegressionDataset(DummyAudioDataset):
        def __getitem__(self, idx: int):
            item = super().__getitem__(idx)
            item["metadata"]["target"] = float(idx * 0.5)
            return item

    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=False)
    dataset = DummyRegressionDataset(num_samples=16, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=8)

    opt_regressor = BayesOptRegressor(
        base_model="linear",
        param_space={"alpha": [0.1, 1.0, 10.0]},
        n_trials=3,
        cv=2,
        show_progress=False,
    )
    task = RegressionTask(regressor=opt_regressor, target_key="target")

    result = benchmark.run(
        model=model,
        model_name="dummy_opt_reg_encoder",
        dataset=dataset,
        task=task,
        batch_size=4,
        device="cpu",
    )

    assert "mse" in result.metrics
    assert "r2" in result.metrics
    assert "best_params" in result.metrics
    assert result.metrics["best_params"]["alpha"] in [0.1, 1.0, 10.0]

    result.print_summary()


def test_bayes_opt_disallowed_aliases():
    for alias in ("bayesopt", "bayesian_optimization", "bayes_optimization"):
        with pytest.raises(ValueError, match="Unsupported classifier type"):
            create_classifier(alias)
        with pytest.raises(ValueError, match="Unsupported regressor type"):
            create_regressor(alias)
