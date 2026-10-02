"""Bayesian optimization wrappers for downstream benchmark models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from audio_benchmark.classifiers.models import BaseClassifier, create_classifier
from audio_benchmark.regressors.models import BaseRegressor, create_regressor
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
            if kind in ("categorical", "cat", "choice"):
                return trial.suggest_categorical(name, spec[1])
            elif kind in ("int", "integer"):
                step = spec[3] if len(spec) > 3 else 1
                return trial.suggest_int(name, int(spec[1]), int(spec[2]), step=step)
            elif kind in ("float",):
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
        if kind in ("categorical", "choice"):
            return trial.suggest_categorical(name, spec["choices"])
        elif kind in ("int", "integer"):
            return trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=spec.get("step", 1)
            )
        elif kind in ("float",):
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )

    return spec


def get_default_classifier_param_space(model_name: str) -> dict[str, Any]:
    """Retrieve default hyperparameter search space for standard classifiers."""
    norm = str(model_name).lower().strip()
    if norm in ("linear", "logistic", "logistic_regression", "linear_classifier"):
        return {"C": (1e-3, 1e2, "log")}
    elif norm in ("ridge", "ridge_classifier"):
        return {"alpha": (1e-3, 1e3, "log")}
    elif norm in ("knn", "k_nearest_neighbors"):
        return {
            "n_neighbors": (1, 15),
            "metric": ["minkowski", "cosine", "euclidean"],
            "weights": ["uniform", "distance"],
        }
    elif norm in ("mlp", "multilayer_perceptron"):
        return {
            "hidden_dim": [32, 64, 128, 256],
            "dropout": (0.0, 0.5),
            "lr": (1e-4, 1e-2, "log"),
            "weight_decay": (1e-6, 1e-2, "log"),
        }
    elif norm in ("transformer", "transformer_encoder"):
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


def get_default_regressor_param_space(model_name: str) -> dict[str, Any]:
    """Retrieve default hyperparameter search space for standard regressors."""
    norm = str(model_name).lower().strip()
    if norm in ("linear", "ridge", "linear_regression"):
        return {"alpha": (1e-3, 1e3, "log")}
    elif norm in ("knn", "k_nearest_neighbors"):
        return {
            "n_neighbors": (1, 15),
            "metric": ["minkowski", "cosine", "euclidean"],
            "weights": ["uniform", "distance"],
        }
    elif norm in ("mlp", "multilayer_perceptron"):
        return {
            "hidden_dim": [32, 64, 128, 256],
            "dropout": (0.0, 0.5),
            "lr": (1e-4, 1e-2, "log"),
            "weight_decay": (1e-6, 1e-2, "log"),
        }
    elif norm in ("transformer", "transformer_encoder"):
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
        base_model: Identifier or instance of the underlying classifier
            ('linear', 'ridge', 'knn', 'mlp', 'transformer', or BaseClassifier instance/class).
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
        base_model: str | BaseClassifier | type[BaseClassifier] = "knn",
        param_space: (
            dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]] | None
        ) = None,
        n_trials: int = 15,
        cv: int = 3,
        metric: str = "accuracy",
        random_state: int | None = 42,
        show_progress: bool = True,
        base_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if "classifier" in kwargs:
            base_model = kwargs.pop("classifier")

        self.base_model = base_model
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.show_progress = show_progress

        merged_base_kwargs = dict(base_kwargs or {})
        merged_base_kwargs.update(kwargs)
        self.base_kwargs = merged_base_kwargs

        # Determine model name identifier
        if isinstance(base_model, str):
            self.model_name = base_model.lower().strip()
        elif isinstance(base_model, type):
            self.model_name = base_model.__name__
        elif hasattr(base_model, "__class__"):
            self.model_name = base_model.__class__.__name__
        else:
            self.model_name = str(base_model)

        # Set parameter space
        if param_space is None and isinstance(base_model, str):
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
        if isinstance(self.base_model, str):
            return self.base_model.lower().strip() in (
                "transformer",
                "transformer_encoder",
                "transformer_classifier",
            )
        return getattr(self.base_model, "accepts_sequence", False)

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample candidate hyperparameters for a trial."""
        if callable(self.param_space):
            return self.param_space(trial)
        return {
            name: _sample_param(trial, name, spec)
            for name, spec in self.param_space.items()
        }

    def _create_model(self, **kwargs) -> BaseClassifier:
        """Instantiate an underlying classifier with the given parameters."""
        if any(name in self.model_name.lower() for name in ("linear", "ridge", "knn")):
            kwargs.pop("show_progress", None)

        if isinstance(self.base_model, str):
            return create_classifier(self.base_model, **kwargs)
        elif isinstance(self.base_model, type) and issubclass(
            self.base_model, BaseClassifier
        ):
            return self.base_model(**kwargs)
        elif isinstance(self.base_model, BaseClassifier):
            return self.base_model.__class__(**kwargs)
        elif callable(self.base_model):
            return self.base_model(**kwargs)
        else:
            raise TypeError(f"Unsupported base_model type: {type(self.base_model)}")

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

                    model = self._create_model(**merged_kwargs)
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
        self.best_model = self._create_model(**best_model_kwargs)
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
        base_model: Identifier or instance of the underlying regressor
            ('linear', 'ridge', 'knn', 'mlp', 'transformer', or BaseRegressor instance/class).
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
        base_model: str | BaseRegressor | type[BaseRegressor] = "knn",
        param_space: (
            dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]] | None
        ) = None,
        n_trials: int = 15,
        cv: int = 3,
        metric: str = "r2",
        random_state: int | None = 42,
        show_progress: bool = True,
        base_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if "regressor" in kwargs:
            base_model = kwargs.pop("regressor")

        self.base_model = base_model
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.show_progress = show_progress

        merged_base_kwargs = dict(base_kwargs or {})
        merged_base_kwargs.update(kwargs)
        self.base_kwargs = merged_base_kwargs

        if isinstance(base_model, str):
            self.model_name = base_model.lower().strip()
        elif isinstance(base_model, type):
            self.model_name = base_model.__name__
        elif hasattr(base_model, "__class__"):
            self.model_name = base_model.__class__.__name__
        else:
            self.model_name = str(base_model)

        if param_space is None and isinstance(base_model, str):
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
        if isinstance(self.base_model, str):
            return self.base_model.lower().strip() in (
                "transformer",
                "transformer_encoder",
                "transformer_regressor",
            )
        return getattr(self.base_model, "accepts_sequence", False)

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample candidate hyperparameters for a trial."""
        if callable(self.param_space):
            return self.param_space(trial)
        return {
            name: _sample_param(trial, name, spec)
            for name, spec in self.param_space.items()
        }

    def _create_model(self, **kwargs) -> BaseRegressor:
        """Instantiate an underlying regressor with the given parameters."""
        if any(name in self.model_name.lower() for name in ("linear", "ridge", "knn")):
            kwargs.pop("show_progress", None)

        if isinstance(self.base_model, str):
            return create_regressor(self.base_model, **kwargs)
        elif isinstance(self.base_model, type) and issubclass(
            self.base_model, BaseRegressor
        ):
            return self.base_model(**kwargs)
        elif isinstance(self.base_model, BaseRegressor):
            return self.base_model.__class__(**kwargs)
        elif callable(self.base_model):
            return self.base_model(**kwargs)
        else:
            raise TypeError(f"Unsupported base_model type: {type(self.base_model)}")

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

        is_minimize = (
            self.metric in ("mse", "rmse", "mae", "loss")
            or "error" in self.metric.lower()
        )
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

                    model = self._create_model(**merged_kwargs)
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
        self.best_model = self._create_model(**best_model_kwargs)
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
