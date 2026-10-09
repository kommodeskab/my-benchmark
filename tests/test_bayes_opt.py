"""Tests for Bayesian Optimization wrappers (BayesOptClassifier, BayesOptRegressor)."""

import numpy as np
import pytest

from audio_benchmark.bayes_opt import (
    BayesOptClassifier,
    BayesOptRegressor,
    _sample_param,
)
from audio_benchmark.benchmark import AudioBenchmark
from audio_benchmark.classifiers import (
    KNNBenchmarkClassifier,
    MLPBenchmarkClassifier,
)
from audio_benchmark.dummy import DummyAudioDataset, DummyAudioModel
from audio_benchmark.regressors import (
    LinearBenchmarkRegressor,
)
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


def test_param_space_validation():
    knn = KNNBenchmarkClassifier()
    reg = LinearBenchmarkRegressor()

    # param_space=None raises ValueError
    with pytest.raises(ValueError, match="param_space cannot be None"):
        BayesOptClassifier(base_model=knn, param_space=None)

    with pytest.raises(ValueError, match="param_space cannot be None"):
        BayesOptRegressor(base_model=reg, param_space=None)

    # Missing param_space raises TypeError
    with pytest.raises(TypeError):
        BayesOptClassifier(base_model=knn)

    with pytest.raises(TypeError):
        BayesOptRegressor(base_model=reg)

    # Invalid param_space type raises TypeError
    with pytest.raises(TypeError, match="param_space must be a dict or a callable"):
        BayesOptClassifier(base_model=knn, param_space=123)

    with pytest.raises(TypeError, match="param_space must be a dict or a callable"):
        BayesOptRegressor(base_model=reg, param_space="invalid")


def test_bayes_opt_classifier_knn(synthetic_classification_data):
    X_train, y_train, X_test, y_test = synthetic_classification_data

    clf = BayesOptClassifier(
        base_model=KNNBenchmarkClassifier(),
        param_space={"n_neighbors": (1, 5), "metric": ["minkowski", "cosine"]},
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
        base_model=KNNBenchmarkClassifier(),
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
        base_model=KNNBenchmarkClassifier(),
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


def test_bayes_opt_preserves_base_model_parameters(synthetic_classification_data):
    X_train, y_train, _, _ = synthetic_classification_data
    base = MLPBenchmarkClassifier(hidden_dim=128, epochs=2, batch_size=16)
    clf = BayesOptClassifier(
        base_model=base,
        param_space={"lr": (1e-4, 1e-2, "log")},
        n_trials=2,
        cv=2,
        show_progress=False,
    )
    clf.fit(X_train, y_train)
    assert clf.best_model.hidden_dim == 128
    assert clf.best_model.epochs == 2
    assert clf.best_model.batch_size == 16


def test_bayes_opt_classifier_string_rejected():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        BayesOptClassifier(base_model="knn", param_space={})
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        BayesOptClassifier(base_model=None, param_space={})
    with pytest.raises(TypeError):
        BayesOptClassifier()


def test_bayes_opt_regressor_linear(synthetic_regression_data):
    X_train, y_train, X_test, y_test = synthetic_regression_data

    reg = BayesOptRegressor(
        base_model=LinearBenchmarkRegressor(),
        param_space={"alpha": (1e-3, 10.0, "log")},
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
        base_model=LinearBenchmarkRegressor(),
        param_space={"alpha": (1e-3, 10.0, "log")},
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


def test_bayes_opt_regressor_string_rejected():
    with pytest.raises(TypeError, match="Passing strings as models is not supported"):
        BayesOptRegressor(base_model="linear", param_space={})
    with pytest.raises(TypeError, match="Passing None as a model is not allowed"):
        BayesOptRegressor(base_model=None, param_space={})
    with pytest.raises(TypeError):
        BayesOptRegressor()


def test_bayes_opt_in_task_and_benchmark(tmp_path):
    dataset = DummyAudioDataset(num_samples=16, audio_length=320, num_classes=2)
    testset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=8)

    opt_classifier = BayesOptClassifier(
        base_model=KNNBenchmarkClassifier(),
        param_space={"n_neighbors": [1, 3]},
        n_trials=3,
        cv=2,
        show_progress=False,
    )
    task = GlobalClassificationTask(classifier=opt_classifier)

    benchmark = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=task,
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=False,
    )

    result = benchmark.run(
        model=model,
        model_name="dummy_opt_encoder",
        batch_size=4,
        device="cpu",
    )

    assert "accuracy" in result.metrics
    assert "best_params" in result.metrics
    assert result.metrics["best_params"]["n_neighbors"] in [1, 3]
    assert "best_cv_score" in result.metrics

    result.print_summary()


def test_bayes_opt_regression_task_and_benchmark(tmp_path):
    class DummyRegressionDataset(DummyAudioDataset):
        def __getitem__(self, idx: int):
            item = super().__getitem__(idx)
            item["metadata"]["target"] = float(idx * 0.5)
            return item

    dataset = DummyRegressionDataset(num_samples=16, audio_length=320, num_classes=2)
    testset = DummyRegressionDataset(num_samples=8, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=8)

    opt_regressor = BayesOptRegressor(
        base_model=LinearBenchmarkRegressor(),
        param_space={"alpha": [0.1, 1.0, 10.0]},
        n_trials=3,
        cv=2,
        show_progress=False,
    )
    task = RegressionTask(regressor=opt_regressor, target_key="target")

    benchmark = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=task,
        dataset_name="dummy_regression",
        cache_dir=tmp_path,
        enable_cache=False,
    )

    result = benchmark.run(
        model=model,
        model_name="dummy_opt_reg_encoder",
        batch_size=4,
        device="cpu",
    )

    assert "mse" in result.metrics
    assert "r2" in result.metrics
    assert "best_params" in result.metrics
    assert result.metrics["best_params"]["alpha"] in [0.1, 1.0, 10.0]

    result.print_summary()


def test_bayes_opt_string_disallowed():
    for alias in ("bayes_opt", "bayesopt", "bayesian_optimization", "knn", "linear"):
        with pytest.raises(
            TypeError, match="Passing strings as models is not supported"
        ):
            BayesOptClassifier(base_model=alias, param_space={})
        with pytest.raises(
            TypeError, match="Passing strings as models is not supported"
        ):
            BayesOptRegressor(base_model=alias, param_space={})
