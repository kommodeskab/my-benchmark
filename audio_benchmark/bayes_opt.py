"""Bayesian optimization wrappers for downstream benchmark models."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from audio_benchmark.classifiers.models import (
    BaseClassifier,
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    MLPBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)
from audio_benchmark.regressors.models import (
    BaseRegressor,
    KNNBenchmarkRegressor,
    LinearBenchmarkRegressor,
    MLPBenchmarkRegressor,
    TransformerBenchmarkRegressor,
)
from audio_benchmark.utils import to_numpy


def _sample_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """Sample a parameter value from an Optuna trial based on a specification.

    Handles discrete (categorical choices, integer ranges) and continuous parameters.

    Args:
        trial: Current Optuna trial.
        name: Name of the hyperparameter.
        spec: Specification defining the search range or choices.

    Returns:
        Sampled parameter value.
    """
    if isinstance(spec, list):
        # Discrete categorical choices
        return trial.suggest_categorical(name, spec)

    if isinstance(spec, tuple):
        if len(spec) > 0 and isinstance(spec[0], str):
            kind = spec[0].lower().strip()
            if kind == "categorical":
                return trial.suggest_categorical(name, spec[1])
            elif kind == "int":
                step = spec[3] if len(spec) > 3 else 1
                return trial.suggest_int(name, int(spec[1]), int(spec[2]), step=step)
            elif kind == "float":
                log = spec[3] if len(spec) > 3 else False
                return trial.suggest_float(
                    name, float(spec[1]), float(spec[2]), log=bool(log)
                )

        if len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                return trial.suggest_int(name, low, high)
            return trial.suggest_float(name, float(low), float(high))
        elif len(spec) == 3:
            low, high, opt = spec
            if opt in ("log", True):
                return trial.suggest_float(name, float(low), float(high), log=True)
            elif opt is False:
                return trial.suggest_float(name, float(low), float(high), log=False)
            elif (
                isinstance(opt, int) and isinstance(low, int) and isinstance(high, int)
            ):
                return trial.suggest_int(name, low, high, step=opt)

    if isinstance(spec, dict):
        kind = spec.get("type", "float").lower()
        if kind == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            return trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=spec.get("step", 1)
            )
        elif kind == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )

    return spec


def get_default_classifier_param_space(
    base_model: BaseClassifier | type[BaseClassifier],
) -> dict[str, Any]:
    """Retrieve default hyperparameter search space for standard classifiers."""
    if base_model is None:
        raise TypeError(
            "base_model cannot be None. Please pass an instantiated BaseClassifier instance or subclass."
        )
    elif isinstance(base_model, str):
        raise TypeError(
            f"Passing strings as models is not supported (got '{base_model}'). "
            "Please pass an instantiated BaseClassifier instance."
        )

    cls = base_model if isinstance(base_model, type) else base_model.__class__
    if not issubclass(cls, BaseClassifier):
        raise TypeError(
            f"base_model must be an instance or subclass of BaseClassifier, got {cls.__name__}."
        )

    if issubclass(cls, LinearBenchmarkClassifier):
        if (
            not isinstance(base_model, type)
            and getattr(base_model, "model_type", None) == "ridge"
        ):
            return {"alpha": (1e-3, 1e3, "log")}
        return {"C": (1e-3, 1e2, "log")}
    elif issubclass(cls, KNNBenchmarkClassifier):
        return {
            "n_neighbors": (1, 15),
            "metric": ["minkowski", "cosine", "euclidean"],
            "weights": ["uniform", "distance"],
        }
    elif issubclass(cls, MLPBenchmarkClassifier):
        return {
            "hidden_dim": [128, 256, 512, 1024],
            "num_layers": [1, 2],
            "dropout": (0.0, 0.5),
            "lr": (1e-4, 1e-2, "log"),
            "weight_decay": (1e-6, 1e-2, "log"),
        }
    elif issubclass(cls, TransformerBenchmarkClassifier):
        return {
            "d_model": [32, 64, 128],
            "nhead": [2, 4],
            "num_layers": (1, 3),
            "dim_feedforward": [64, 128, 256],
            "dropout": (0.0, 0.3),
            "lr": (1e-4, 1e-2, "log"),
            "aggregation": ["cls", "mean"],
        }
    return {}


def get_default_regressor_param_space(
    base_model: BaseRegressor | type[BaseRegressor],
) -> dict[str, Any]:
    """Retrieve default hyperparameter search space for standard regressors."""
    if base_model is None:
        raise TypeError(
            "base_model cannot be None. Please pass an instantiated BaseRegressor instance or subclass."
        )
    elif isinstance(base_model, str):
        raise TypeError(
            f"Passing strings as models is not supported (got '{base_model}'). "
            "Please pass an instantiated BaseRegressor instance."
        )

    cls = base_model if isinstance(base_model, type) else base_model.__class__
    if not issubclass(cls, BaseRegressor):
        raise TypeError(
            f"base_model must be an instance or subclass of BaseRegressor, got {cls.__name__}."
        )

    if issubclass(cls, LinearBenchmarkRegressor):
        return {"alpha": (1e-3, 1e3, "log")}
    elif issubclass(cls, KNNBenchmarkRegressor):
        return {
            "n_neighbors": (1, 15),
            "metric": ["minkowski", "cosine", "euclidean"],
            "weights": ["uniform", "distance"],
        }
    elif issubclass(cls, MLPBenchmarkRegressor):
        return {
            "hidden_dim": [128, 256, 512, 1024],
            "num_layers": [1, 2],
            "dropout": (0.0, 0.5),
            "lr": (1e-4, 1e-2, "log"),
            "weight_decay": (1e-6, 1e-2, "log"),
        }
    elif issubclass(cls, TransformerBenchmarkRegressor):
        return {
            "d_model": [32, 64, 128],
            "nhead": [2, 4],
            "num_layers": (1, 3),
            "dim_feedforward": [64, 128, 256],
            "dropout": (0.0, 0.3),
            "lr": (1e-4, 1e-2, "log"),
            "aggregation": ["cls", "mean"],
        }
    return {}


class BayesOptClassifier(BaseClassifier):
    """Bayesian optimization wrapper for downstream benchmark classifiers.

    Optimizes hyperparameters over the training set using K-fold cross-validation
    with Optuna, and fits the best-performing model configuration.

    Args:
        base_model: Instantiated BaseClassifier instance to optimize.
        param_space: Custom search space dict or callable taking `trial` and returning a param dict.
            If None, the default search space for `base_model` is used.
        n_trials: Number of Bayesian optimization trials to run.
        cv: Number of cross-validation folds.
        metric: Evaluation metric to maximize ('accuracy', 'macro_f1', 'micro_f1').
        random_state: Random seed for sampling and cross-validation splits.
        show_progress: Whether to display a tqdm progress bar during optimization.
        base_kwargs: Additional fixed keyword arguments passed to the underlying classifier.
    """

    def __init__(
        self,
        base_model: BaseClassifier,
        param_space: (
            dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]] | None
        ) = None,
        n_trials: int = 15,
        cv: int = 3,
        metric: str = "accuracy",
        random_state: int | None = 42,
        show_progress: bool = True,
        base_kwargs: dict[str, Any] | None = None,
    ):
        if base_model is None:
            raise TypeError(
                "base_model must be an instantiated BaseClassifier instance, got None. "
                "Passing None as a model is not allowed."
            )
        elif isinstance(base_model, str):
            raise TypeError(
                f"Passing strings as models is not supported (got '{base_model}'). "
                "Please pass an instantiated BaseClassifier instance."
            )
        elif not isinstance(base_model, BaseClassifier):
            raise TypeError(
                f"base_model must be an instantiated BaseClassifier instance, got {type(base_model).__name__}."
            )

        self.base_model = base_model
        self.model_name = base_model.__class__.__name__
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.show_progress = show_progress
        self.base_kwargs = dict(base_kwargs or {})

        # Set parameter space
        if param_space is None:
            self.param_space = get_default_classifier_param_space(base_model)
        else:
            self.param_space = param_space or {}

        self.best_model: BaseClassifier | None = None
        self.best_params: dict[str, Any] = {}
        self.best_score: float | None = None
        self.study: optuna.Study | None = None

    @property
    def accepts_sequence(self) -> bool:
        """Indicate whether the underlying model accepts unpooled sequence inputs."""
        if self.best_model is not None:
            return getattr(self.best_model, "accepts_sequence", False)
        return getattr(self.base_model, "accepts_sequence", False)

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample candidate hyperparameters for a trial."""
        if callable(self.param_space):
            return self.param_space(trial)
        return {
            name: _sample_param(trial, name, spec)
            for name, spec in self.param_space.items()
        }

    def _create_model(
        self, model_kwargs: dict[str, Any] | None = None
    ) -> BaseClassifier:
        """Instantiate an underlying classifier with the given parameters."""
        sig = inspect.signature(self.base_model.__class__.__init__)
        init_kwargs = {}
        for p in sig.parameters:
            if p in ("self", "args", "kwargs"):
                continue
            attr = (
                "device_str"
                if p == "device" and hasattr(self.base_model, "device_str")
                else p
            )
            if hasattr(self.base_model, attr):
                init_kwargs[p] = getattr(self.base_model, attr)

        init_kwargs.update(self.base_kwargs)
        if model_kwargs:
            init_kwargs.update(model_kwargs)

        if not any(p == "show_progress" for p in sig.parameters):
            init_kwargs.pop("show_progress", None)

        return self.base_model.__class__(**init_kwargs)

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> BayesOptClassifier:
        """Run Bayesian optimization across candidate hyperparameters and fit the best model.

        Args:
            X: Input features or sequence representations.
            y: Target class labels.

        Returns:
            BayesOptClassifier: Fitted wrapper instance containing the best model.
        """
        X_data = X
        y_np = to_numpy(y).ravel()

        if len(y_np) < 2:
            raise ValueError("Cannot run cross-validation with fewer than 2 samples.")

        # Determine actual CV folds safely based on sample counts and class counts
        _, counts = np.unique(y_np, return_counts=True)
        min_count = int(np.min(counts)) if len(counts) > 0 else 0
        actual_cv = max(2, min(self.cv, len(y_np)))
        if min_count >= actual_cv:
            cv_splitter = StratifiedKFold(
                n_splits=actual_cv, shuffle=True, random_state=self.random_state
            )
        else:
            cv_splitter = KFold(
                n_splits=actual_cv, shuffle=True, random_state=self.random_state
            )

        def objective(trial: optuna.Trial) -> float:
            try:
                sampled = self._sample_params(trial)
                merged_kwargs = {**self.base_kwargs, **sampled}
                # Suppress progress bar during trial CV to prevent terminal clutter
                if "show_progress" not in merged_kwargs:
                    merged_kwargs["show_progress"] = False

                scores = []
                for train_idx, val_idx in cv_splitter.split(X_data, y_np):
                    X_tr = X_data[train_idx]
                    y_tr = y_np[train_idx]
                    X_va = X_data[val_idx]
                    y_va = y_np[val_idx]

                    model = self._create_model(merged_kwargs)
                    model.fit(X_tr, y_tr)
                    metrics = model.evaluate(X_va, y_va)
                    scores.append(float(metrics.get(self.metric, 0.0)))

                return float(np.mean(scores))
            except (ValueError, TypeError, RuntimeError, optuna.TrialPruned):
                return float("-inf")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        pbar = tqdm(
            total=self.n_trials,
            desc=f"BayesOpt [{self.model_name}]",
            disable=not self.show_progress,
        )
        for _ in range(self.n_trials):
            self.study.optimize(objective, n_trials=1)
            pbar.update(1)
            pbar.set_postfix({self.metric: f"{self.study.best_value:.4f}"})
        pbar.close()

        self.best_params = dict(self.study.best_params)
        self.best_score = float(self.study.best_value)

        # Train the best-performing model on the full training set
        best_model_kwargs = {**self.base_kwargs, **self.best_params}
        if "show_progress" not in best_model_kwargs:
            best_model_kwargs["show_progress"] = self.show_progress
        self.best_model = self._create_model(best_model_kwargs)
        self.best_model.fit(X, y)

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class labels using the best-performing fitted model."""
        if self.best_model is None:
            raise RuntimeError("BayesOptClassifier has not been fitted yet.")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class probabilities using the best-performing fitted model."""
        if self.best_model is None:
            raise RuntimeError("BayesOptClassifier has not been fitted yet.")
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        raise AttributeError(
            f"{self.best_model.__class__.__name__} does not support predict_proba."
        )

    def evaluate(
        self, X_test: np.ndarray | torch.Tensor, y_test: np.ndarray | torch.Tensor
    ) -> dict[str, float]:
        """Evaluate the best model on test features and ground truth labels."""
        if self.best_model is None:
            raise RuntimeError("BayesOptClassifier has not been fitted yet.")
        return self.best_model.evaluate(X_test, y_test)


class BayesOptRegressor(BaseRegressor):
    """Bayesian optimization wrapper for downstream benchmark regressors.

    Optimizes hyperparameters over the training set using K-fold cross-validation
    with Optuna, and fits the best-performing model configuration.

    Args:
        base_model: Instantiated BaseRegressor instance to optimize.
        param_space: Custom search space dict or callable taking `trial` and returning a param dict.
            If None, the default search space for `base_model` is used.
        n_trials: Number of Bayesian optimization trials to run.
        cv: Number of cross-validation folds.
        metric: Evaluation metric to optimize ('r2', 'mse', 'rmse', 'mae', 'pearson_r').
        random_state: Random seed for sampling and cross-validation splits.
        show_progress: Whether to display a tqdm progress bar during optimization.
        base_kwargs: Additional fixed keyword arguments passed to the underlying regressor.
    """

    def __init__(
        self,
        base_model: BaseRegressor,
        param_space: (
            dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]] | None
        ) = None,
        n_trials: int = 15,
        cv: int = 3,
        metric: str = "r2",
        random_state: int | None = 42,
        show_progress: bool = True,
        base_kwargs: dict[str, Any] | None = None,
    ):
        if base_model is None:
            raise TypeError(
                "base_model must be an instantiated BaseRegressor instance, got None. "
                "Passing None as a model is not allowed."
            )
        elif isinstance(base_model, str):
            raise TypeError(
                f"Passing strings as models is not supported (got '{base_model}'). "
                "Please pass an instantiated BaseRegressor instance."
            )
        elif not isinstance(base_model, BaseRegressor):
            raise TypeError(
                f"base_model must be an instantiated BaseRegressor instance, got {type(base_model).__name__}."
            )

        self.base_model = base_model
        self.model_name = base_model.__class__.__name__
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.show_progress = show_progress
        self.base_kwargs = dict(base_kwargs or {})

        # Set parameter space
        if param_space is None:
            self.param_space = get_default_regressor_param_space(base_model)
        else:
            self.param_space = param_space or {}

        self.best_model: BaseRegressor | None = None
        self.best_params: dict[str, Any] = {}
        self.best_score: float | None = None
        self.study: optuna.Study | None = None

    @property
    def accepts_sequence(self) -> bool:
        """Indicate whether the underlying model accepts unpooled sequence inputs."""
        if self.best_model is not None:
            return getattr(self.best_model, "accepts_sequence", False)
        return getattr(self.base_model, "accepts_sequence", False)

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample candidate hyperparameters for a trial."""
        if callable(self.param_space):
            return self.param_space(trial)
        return {
            name: _sample_param(trial, name, spec)
            for name, spec in self.param_space.items()
        }

    def _create_model(
        self, model_kwargs: dict[str, Any] | None = None
    ) -> BaseRegressor:
        """Instantiate an underlying regressor with the given parameters."""
        sig = inspect.signature(self.base_model.__class__.__init__)
        init_kwargs = {}
        for p in sig.parameters:
            if p in ("self", "args", "kwargs"):
                continue
            attr = (
                "device_str"
                if p == "device" and hasattr(self.base_model, "device_str")
                else p
            )
            if hasattr(self.base_model, attr):
                init_kwargs[p] = getattr(self.base_model, attr)

        init_kwargs.update(self.base_kwargs)
        if model_kwargs:
            init_kwargs.update(model_kwargs)

        if not any(p == "show_progress" for p in sig.parameters):
            init_kwargs.pop("show_progress", None)

        return self.base_model.__class__(**init_kwargs)

    def fit(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> BayesOptRegressor:
        """Run Bayesian optimization across candidate hyperparameters and fit the best model.

        Args:
            X: Input features or sequence representations.
            y: Target continuous values.

        Returns:
            BayesOptRegressor: Fitted wrapper instance containing the best model.
        """
        X_data = X
        y_np = to_numpy(y).ravel()

        if len(y_np) < 2:
            raise ValueError("Cannot run cross-validation with fewer than 2 samples.")

        actual_cv = max(2, min(self.cv, len(y_np)))
        cv_splitter = KFold(
            n_splits=actual_cv, shuffle=True, random_state=self.random_state
        )

        is_minimize = self.metric in ("mse", "rmse", "mae")
        direction = "minimize" if is_minimize else "maximize"

        def objective(trial: optuna.Trial) -> float:
            try:
                sampled = self._sample_params(trial)
                merged_kwargs = {**self.base_kwargs, **sampled}
                if "show_progress" not in merged_kwargs:
                    merged_kwargs["show_progress"] = False

                scores = []
                for train_idx, val_idx in cv_splitter.split(X_data, y_np):
                    X_tr = X_data[train_idx]
                    y_tr = y_np[train_idx]
                    X_va = X_data[val_idx]
                    y_va = y_np[val_idx]

                    model = self._create_model(merged_kwargs)
                    model.fit(X_tr, y_tr)
                    metrics = model.evaluate(X_va, y_va)
                    scores.append(float(metrics.get(self.metric, 0.0)))

                return float(np.mean(scores))
            except (ValueError, TypeError, RuntimeError, optuna.TrialPruned):
                return float("inf") if is_minimize else float("-inf")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction=direction, sampler=sampler)

        pbar = tqdm(
            total=self.n_trials,
            desc=f"BayesOpt [{self.model_name}]",
            disable=not self.show_progress,
        )
        for _ in range(self.n_trials):
            self.study.optimize(objective, n_trials=1)
            pbar.update(1)
            pbar.set_postfix({self.metric: f"{self.study.best_value:.4f}"})
        pbar.close()

        self.best_params = dict(self.study.best_params)
        self.best_score = float(self.study.best_value)

        # Train the best-performing model on the full training set
        best_model_kwargs = {**self.base_kwargs, **self.best_params}
        if "show_progress" not in best_model_kwargs:
            best_model_kwargs["show_progress"] = self.show_progress
        self.best_model = self._create_model(best_model_kwargs)
        self.best_model.fit(X, y)

        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict continuous values using the best-performing fitted model."""
        if self.best_model is None:
            raise RuntimeError("BayesOptRegressor has not been fitted yet.")
        return self.best_model.predict(X)

    def evaluate(
        self, X_test: np.ndarray | torch.Tensor, y_test: np.ndarray | torch.Tensor
    ) -> dict[str, float]:
        """Evaluate the best model on test features and continuous targets."""
        if self.best_model is None:
            raise RuntimeError("BayesOptRegressor has not been fitted yet.")
        return self.best_model.evaluate(X_test, y_test)
