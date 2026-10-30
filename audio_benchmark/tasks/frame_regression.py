"""Frame-level audio regression task."""

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset

from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.regressors.models import BaseRegressor
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.downsampling import (
    align_features_to_targets,
    parse_frame_targets,
)


class FrameRegressionTask(BaseTask):
    """Frame-level audio regression task.

    Evaluates frame-level audio regression by linearly interpolating representation
    features to match the sequence length of raw continuous targets.
    Raw ground-truth targets are never modified.
    """

    def __init__(
        self,
        regressor: BaseRegressor,
        name: str = "frame_regression",
        target_key: str = "frame_targets",
        metric: str = "r2",
        pooling: str = "linear",
    ):
        """Initialize the FrameRegressionTask.

        Args:
            regressor: Downstream regressor instance conforming to BaseRegressor.
            name: Identifier for the task.
            target_key: Metadata key containing per-frame continuous targets (flat torch.Tensor or list).
            metric: Primary metric for the task ('r2', 'mse', 'rmse', 'mae', 'pearson_r').
            pooling: Temporal alignment strategy ('linear' interpolation).
        """
        super().__init__(name=name)
        valid_metrics = ("r2", "mse", "rmse", "mae", "pearson_r")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        valid_pooling = ("linear",)
        if pooling not in valid_pooling:
            raise ValueError(
                f"Unknown pooling '{pooling}'. Supported pooling: {valid_pooling}"
            )

        self.metric = metric
        self.target_key = target_key
        self.pooling = pooling

        if not isinstance(regressor, BaseRegressor):
            raise TypeError(
                f"regressor must be an instantiated BaseRegressor instance, got {type(regressor).__name__}."
            )
        self.regressor = regressor

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains valid per-frame continuous targets."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        parsed = parse_dataset_item(dataset[0])
        targets = parsed["metadata"][self.target_key]
        values = parse_frame_targets(
            targets,
            key_name=self.target_key,
        )
        for idx, val in enumerate(values):
            try:
                float(val)
            except (ValueError, TypeError) as e:
                raise InvalidDatasetError(
                    f"Frame target at index {idx} in '{self.target_key}' cannot be converted to float: {val!r}."
                ) from e

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Align representation features to raw frame targets and evaluate regressor.

        Ground-truth targets are never downsampled or modified. Features are aligned
        to targets via 1D linear interpolation.

        Args:
            train_representations: Tensor of shape (N_train, T', C') or sequence of representations.
            train_metadata: List of metadata dicts corresponding to each train clip.
            test_representations: Tensor of shape (N_test, T', C') or sequence of representations.
            test_metadata: List of metadata dicts corresponding to each test clip.

        Returns:
            dict[str, Any]: Frame-level regression metrics.
        """
        if len(train_representations) != len(train_metadata):
            raise TaskError(
                f"Length mismatch: {len(train_representations)} train representations vs {len(train_metadata)} train metadata entries."
            )
        if len(test_representations) != len(test_metadata):
            raise TaskError(
                f"Length mismatch: {len(test_representations)} test representations vs {len(test_metadata)} test metadata entries."
            )
        if len(train_representations) == 0:
            raise TaskError("No train representations provided for evaluation.")
        if len(test_representations) == 0:
            raise TaskError("No test representations provided for evaluation.")

        train_aligned: list[torch.Tensor] = []
        train_targets_list: list[list[float]] = []
        for idx, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            raw_target = meta[self.target_key]
            raw_targets = parse_frame_targets(
                raw_target,
                key_name=self.target_key,
            )
            clip_reps = train_representations[idx]
            aligned = align_features_to_targets(
                representations=clip_reps,
                target_len=len(raw_targets),
            )
            train_aligned.append(aligned)
            try:
                train_targets_list.append([float(t) for t in raw_targets])
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Train sample at index {idx} frame target cannot be converted to float: {e}"
                ) from e

        test_aligned: list[torch.Tensor] = []
        test_targets_list: list[list[float]] = []
        for idx, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            raw_target = meta[self.target_key]
            raw_targets = parse_frame_targets(
                raw_target,
                key_name=self.target_key,
            )
            clip_reps = test_representations[idx]
            aligned = align_features_to_targets(
                representations=clip_reps,
                target_len=len(raw_targets),
            )
            test_aligned.append(aligned)
            try:
                test_targets_list.append([float(t) for t in raw_targets])
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Test sample at index {idx} frame target cannot be converted to float: {e}"
                ) from e

        X_train = torch.cat(train_aligned, dim=0)
        X_test = torch.cat(test_aligned, dim=0)

        train_flat = [t for sub in train_targets_list for t in sub]
        test_flat = [t for sub in test_targets_list for t in sub]

        y_train = torch.as_tensor(train_flat, dtype=torch.float32)
        y_test = torch.as_tensor(test_flat, dtype=torch.float32)

        regressor_inst = self.regressor
        regressor_name = regressor_inst.__class__.__name__

        start_time = time.perf_counter()
        regressor_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = regressor_inst.evaluate(X_test, y_test)
        train_metrics = regressor_inst.evaluate(X_train, y_train)

        results = {
            "task": self.name,
            "regressor": regressor_name,
            "value": float(metrics[self.metric]),
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson_r"],
            "train_mse": train_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "training_time": training_time,
        }

        return results

    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of frame regression.

        Validates the datasets, computes frame target distributions across train and test sets,
        and computes a trivial baseline (frame target mean predictor) expected metric.

        Args:
            dataset: Audio dataset to examine for training.
            testset: Optional audio dataset to examine for testing.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be an instance of Dataset, got {type(dataset).__name__}."
            )
        if testset is not None and not isinstance(testset, Dataset):
            raise TypeError(
                f"testset must be an instance of Dataset, got {type(testset).__name__}."
            )
        self.validate_dataset(dataset)
        if testset is not None:
            self.validate_dataset(testset)

        train_metadata = self.extract_metadata(dataset)
        if testset is not None:
            test_metadata = self.extract_metadata(testset)
        else:
            test_metadata = train_metadata

        if len(train_metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        train_targets: list[float] = []
        for idx, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            raw_values = parse_frame_targets(
                meta[self.target_key],
                key_name=self.target_key,
            )
            for t in raw_values:
                try:
                    train_targets.append(float(t))
                except (ValueError, TypeError) as e:
                    raise TaskError(
                        f"Sample at index {idx} frame target cannot be converted to float: {e}"
                    ) from e

        test_targets: list[float] = []
        for idx, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            raw_values = parse_frame_targets(
                meta[self.target_key],
                key_name=self.target_key,
            )
            for t in raw_values:
                try:
                    test_targets.append(float(t))
                except (ValueError, TypeError) as e:
                    raise TaskError(
                        f"Sample at index {idx} frame target cannot be converted to float: {e}"
                    ) from e

        train_targets_np = np.array(train_targets, dtype=np.float32)
        test_targets_np = np.array(test_targets, dtype=np.float32)
        all_targets = (
            np.concatenate([train_targets_np, test_targets_np])
            if testset is not None
            else train_targets_np
        )

        sample_targets = parse_frame_targets(
            train_metadata[0][self.target_key],
            key_name=self.target_key,
        )
        frame_len_sample = len(sample_targets) if train_metadata else 0

        train_mean = float(np.mean(train_targets_np))
        y_pred_baseline = np.full_like(test_targets_np, train_mean)

        baseline_mse = float(mean_squared_error(test_targets_np, y_pred_baseline))
        baseline_rmse = float(np.sqrt(baseline_mse))
        baseline_mae = float(mean_absolute_error(test_targets_np, y_pred_baseline))
        try:
            baseline_r2 = float(r2_score(test_targets_np, y_pred_baseline))
        except (ValueError, ZeroDivisionError):
            baseline_r2 = 0.0

        if self.metric == "r2":
            baseline_value = baseline_r2
        elif self.metric == "mse":
            baseline_value = baseline_mse
        elif self.metric == "rmse":
            baseline_value = baseline_rmse
        elif self.metric == "mae":
            baseline_value = baseline_mae
        elif self.metric == "pearson_r":
            baseline_value = 0.0
        else:
            baseline_value = baseline_r2

        return {
            "task": self.name,
            "total_clips": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_clips": len(train_metadata),
            "test_clips": len(test_metadata) if testset is not None else 0,
            "frame_length": int(frame_len_sample),
            "total_frames": len(train_targets)
            + (len(test_targets) if testset is not None else 0),
            "train_frames": len(train_targets),
            "test_frames": len(test_targets) if testset is not None else 0,
            "target_key": self.target_key,
            "target_min": float(np.min(all_targets)),
            "target_max": float(np.max(all_targets)),
            "target_mean": float(np.mean(all_targets)),
            "target_std": float(np.std(all_targets)),
            "baseline_strategy": "mean_predictor",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }
