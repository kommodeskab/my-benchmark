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

from audio_benchmark.classifiers.models import BaseClassifier
from audio_benchmark.regressors.models import BaseRegressor
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

    return spec


def _clone_model_with_params(
    base_model: Any, model_kwargs: dict[str, Any] | None = None
) -> Any:
    """Instantiate a new model of the same type with updated parameters.

    Args:
        base_model: Instantiated base model instance to clone.
        model_kwargs: Hyperparameters to override on the cloned instance.

    Returns:
        New model instance with updated parameters.
    """
    sig = inspect.signature(base_model.__class__.__init__)
    init_kwargs = {}
    for p in sig.parameters:
        if p in ("self", "args", "kwargs"):
            continue
        attr = (
            "device_str" if p == "device" and hasattr(base_model, "device_str") else p
        )
        if hasattr(base_model, attr):
            init_kwargs[p] = getattr(base_model, attr)

    if model_kwargs:
        init_kwargs.update(model_kwargs)

    if not any(p == "show_progress" for p in sig.parameters):
        init_kwargs.pop("show_progress", None)

    return base_model.__class__(**init_kwargs)


def _run_bayes_opt(
    base_model: Any,
    param_space: dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]],
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    n_trials: int,
    cv: int,
    metric: str,
    random_state: int | None,
    show_progress: bool,
    is_classification: bool,
) -> tuple[Any, dict[str, Any], float, optuna.Study]:
    """Execute Bayesian optimization across cross-validation folds.

    Args:
        base_model: Instantiated base model to optimize.
        param_space: Hyperparameter search space specification or sampler callable.
        X: Input features or sequence representations.
        y: Target values or class labels.
        n_trials: Number of trials to run.
        cv: Number of cross-validation folds.
        metric: Evaluation metric to optimize.
        random_state: Seed for reproducibility.
        show_progress: Whether to show tqdm progress bar.
        is_classification: Whether this is classification (for stratified splitting and direction).

    Returns:
        Tuple of (best_fitted_model, best_params, best_score, study).
    """
    X_data = X
    y_np = to_numpy(y).ravel()

    if len(y_np) < 2:
        raise ValueError("Cannot run cross-validation with fewer than 2 samples.")

    actual_cv = max(2, min(cv, len(y_np)))
    if is_classification:
        _, counts = np.unique(y_np, return_counts=True)
        min_count = int(np.min(counts)) if len(counts) > 0 else 0
        if min_count >= actual_cv:
            cv_splitter = StratifiedKFold(
                n_splits=actual_cv, shuffle=True, random_state=random_state
            )
        else:
            cv_splitter = KFold(
                n_splits=actual_cv, shuffle=True, random_state=random_state
            )
        direction = "maximize"
        failure_score = float("-inf")
    else:
        cv_splitter = KFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
        is_minimize = metric in ("mse", "rmse", "mae")
        direction = "minimize" if is_minimize else "maximize"
        failure_score = float("inf") if is_minimize else float("-inf")

    def sample_fn(trial: optuna.Trial) -> dict[str, Any]:
        if callable(param_space):
            return param_space(trial)
        return {
            name: _sample_param(trial, name, spec) for name, spec in param_space.items()
        }

    def objective(trial: optuna.Trial) -> float:
        try:
            sampled = sample_fn(trial)
            if "show_progress" not in sampled:
                sampled["show_progress"] = False

            scores = []
            for train_idx, val_idx in cv_splitter.split(X_data, y_np):
                X_tr = X_data[train_idx]
                y_tr = y_np[train_idx]
                X_va = X_data[val_idx]
                y_va = y_np[val_idx]

                model = _clone_model_with_params(base_model, sampled)
                model.fit(X_tr, y_tr)
                metrics = model.evaluate(X_va, y_va)
                scores.append(float(metrics.get(metric, 0.0)))

            return float(np.mean(scores))
        except (ValueError, TypeError, RuntimeError, optuna.TrialPruned):
            return failure_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)

    model_name = base_model.__class__.__name__
    pbar = tqdm(
        total=n_trials,
        desc=f"BayesOpt [{model_name}]",
        disable=not show_progress,
    )
    for _ in range(n_trials):
        study.optimize(objective, n_trials=1)
        pbar.update(1)
        pbar.set_postfix({metric: f"{study.best_value:.4f}"})
    pbar.close()

    best_params = dict(study.best_params)
    best_score = float(study.best_value)

    best_model_kwargs = dict(best_params)
    if "show_progress" not in best_model_kwargs:
        best_model_kwargs["show_progress"] = show_progress
    best_model = _clone_model_with_params(base_model, best_model_kwargs)
    best_model.fit(X, y)

    return best_model, best_params, best_score, study


class BayesOptClassifier(BaseClassifier):
    """Bayesian optimization wrapper for downstream benchmark classifiers.

    Optimizes hyperparameters over the training set using K-fold cross-validation
    with Optuna, and fits the best-performing model configuration.

    Args:
        base_model: Instantiated BaseClassifier instance to optimize.
        param_space: Hyperparameter search space specification, provided either as a dictionary
            mapping parameter names to search specifications, or as a callable taking an Optuna
            `trial` and returning a dictionary of sampled parameters.

            Supported dictionary specification formats:
            - Discrete categorical choices: A list of options, e.g. `["minkowski", "cosine"]`
              or `[128, 256, 512]`.
            - Continuous float uniform range: A 2-tuple `(low, high)`, e.g. `(0.0, 0.5)`.
            - Continuous float log-uniform range: A 3-tuple `(low, high, "log")` or `(low, high, True)`,
              e.g. `(1e-4, 1e-2, "log")`.
            - Integer range: A 2-tuple of ints `(low, high)`, e.g. `(1, 15)`, or a 3-tuple with step
              `(low, high, step)`, e.g. `(10, 100, 10)`.
            - Explicit typed tuple: `("categorical", choices)`, `("int", low, high, [step])`, or
              `("float", low, high, [log])`.
            - Callable: A function `def param_space(trial: optuna.Trial) -> dict[str, Any]` that
              samples parameters using Optuna trial methods (e.g., `trial.suggest_float(...)`).
        n_trials: Number of Bayesian optimization trials to run.
        cv: Number of cross-validation folds.
        metric: Evaluation metric to maximize ('accuracy', 'macro_f1', 'micro_f1').
        random_state: Random seed for sampling and cross-validation splits.
        show_progress: Whether to display a tqdm progress bar during optimization.

    Raises:
        TypeError: If `base_model` is not an instantiated BaseClassifier instance, or if `param_space`
            is not a dictionary or callable.
        ValueError: If `param_space` is None.
    """

    def __init__(
        self,
        base_model: BaseClassifier,
        param_space: dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]],
        n_trials: int = 15,
        cv: int = 3,
        metric: str = "accuracy",
        random_state: int | None = 42,
        show_progress: bool = True,
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

        if param_space is None:
            raise ValueError(
                "param_space cannot be None. Explicitly specify the hyperparameter search space "
                "as a dictionary or callable."
            )
        elif not (isinstance(param_space, dict) or callable(param_space)):
            raise TypeError(
                f"param_space must be a dict or a callable, got {type(param_space).__name__}."
            )

        self.base_model = base_model
        self.model_name = base_model.__class__.__name__
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.show_progress = show_progress

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
        self.best_model, self.best_params, self.best_score, self.study = _run_bayes_opt(
            base_model=self.base_model,
            param_space=self.param_space,
            X=X,
            y=y,
            n_trials=self.n_trials,
            cv=self.cv,
            metric=self.metric,
            random_state=self.random_state,
            show_progress=self.show_progress,
            is_classification=True,
        )
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
        param_space: Hyperparameter search space specification, provided either as a dictionary
            mapping parameter names to search specifications, or as a callable taking an Optuna
            `trial` and returning a dictionary of sampled parameters.

            Supported dictionary specification formats:
            - Discrete categorical choices: A list of options, e.g. `["uniform", "distance"]`
              or `[64, 128, 256]`.
            - Continuous float uniform range: A 2-tuple `(low, high)`, e.g. `(0.0, 0.5)`.
            - Continuous float log-uniform range: A 3-tuple `(low, high, "log")` or `(low, high, True)`,
              e.g. `(1e-3, 1e3, "log")`.
            - Integer range: A 2-tuple of ints `(low, high)`, e.g. `(1, 15)`, or a 3-tuple with step
              `(low, high, step)`, e.g. `(10, 100, 10)`.
            - Explicit typed tuple: `("categorical", choices)`, `("int", low, high, [step])`, or
              `("float", low, high, [log])`.
            - Callable: A function `def param_space(trial: optuna.Trial) -> dict[str, Any]` that
              samples parameters using Optuna trial methods (e.g., `trial.suggest_float(...)`).
        n_trials: Number of Bayesian optimization trials to run.
        cv: Number of cross-validation folds.
        metric: Evaluation metric to optimize ('r2', 'mse', 'rmse', 'mae', 'pearson_r').
        random_state: Random seed for sampling and cross-validation splits.
        show_progress: Whether to display a tqdm progress bar during optimization.

    Raises:
        TypeError: If `base_model` is not an instantiated BaseRegressor instance, or if `param_space`
            is not a dictionary or callable.
        ValueError: If `param_space` is None.
    """

    def __init__(
        self,
        base_model: BaseRegressor,
        param_space: dict[str, Any] | Callable[[optuna.Trial], dict[str, Any]],
        n_trials: int = 15,
        cv: int = 3,
        metric: str = "r2",
        random_state: int | None = 42,
        show_progress: bool = True,
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

        if param_space is None:
            raise ValueError(
                "param_space cannot be None. Explicitly specify the hyperparameter search space "
                "as a dictionary or callable."
            )
        elif not (isinstance(param_space, dict) or callable(param_space)):
            raise TypeError(
                f"param_space must be a dict or a callable, got {type(param_space).__name__}."
            )

        self.base_model = base_model
        self.model_name = base_model.__class__.__name__
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.random_state = random_state
        self.show_progress = show_progress

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
        self.best_model, self.best_params, self.best_score, self.study = _run_bayes_opt(
            base_model=self.base_model,
            param_space=self.param_space,
            X=X,
            y=y,
            n_trials=self.n_trials,
            cv=self.cv,
            metric=self.metric,
            random_state=self.random_state,
            show_progress=self.show_progress,
            is_classification=False,
        )
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
