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
    ):
        """Initialize the FrameRegressionTask.

        Args:
            regressor: Downstream regressor instance conforming to BaseRegressor.
            name: Identifier for the task.
            target_key: Metadata key containing per-frame continuous targets (flat torch.Tensor or list).

        Raises:
            TypeError: If regressor is not an instance of BaseRegressor.
        """
        super().__init__(name=name)
        if not isinstance(regressor, BaseRegressor):
            raise TypeError(
                f"regressor must be an instantiated BaseRegressor instance, got {type(regressor).__name__}."
            )

        self.regressor = regressor
        self.target_key = target_key
        self.metric = "r2"
        self.m_min = 0.0
        self.m_max = 1.0

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains valid per-frame continuous targets.

        Args:
            dataset: PyTorch dataset to validate.
        """
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        parsed = parse_dataset_item(dataset[0])
        values = parse_frame_targets(
            parsed["metadata"][self.target_key],
            key_name=self.target_key,
        )
        for idx, val in enumerate(values):
            try:
                float(val)
            except (ValueError, TypeError) as e:
                raise InvalidDatasetError(
                    f"Frame target at index {idx} in '{self.target_key}' cannot be converted to float: {val!r}."
                ) from e

    def _parse_clip_targets(
        self,
        meta: dict[str, Any],
        idx: int,
        prefix: str = "Sample",
    ) -> list[float]:
        """Extract and convert frame targets for a single clip to floats."""
        if self.target_key not in meta:
            raise InvalidDatasetError(
                f"{prefix} at index {idx} metadata does not contain '{self.target_key}'."
            )
        raw_targets = parse_frame_targets(
            meta[self.target_key],
            key_name=self.target_key,
        )
        try:
            return [float(t) for t in raw_targets]
        except (ValueError, TypeError) as e:
            raise TaskError(
                f"{prefix} at index {idx} frame target cannot be converted to float: {e}"
            ) from e

    def _prepare_split(
        self,
        representations: Any,
        metadata: list[dict[str, Any]],
        split_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align representations to frame targets for a split."""
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} {split_name} representations vs "
                f"{len(metadata)} {split_name} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError(f"No {split_name} representations provided for evaluation.")

        prefix = f"{split_name.capitalize()} sample"
        aligned_list: list[torch.Tensor] = []
        flat_targets: list[float] = []
        for idx, meta in enumerate(metadata):
            clip_targets = self._parse_clip_targets(meta, idx, prefix=prefix)
            clip_reps = representations[idx]
            if "rep_length" in meta:
                clip_reps = clip_reps[: int(meta["rep_length"])]
            aligned_list.append(
                align_features_to_targets(
                    representations=clip_reps,
                    target_len=len(clip_targets),
                )
            )
            flat_targets.extend(clip_targets)

        X = torch.cat(aligned_list, dim=0)
        y = torch.as_tensor(flat_targets, dtype=torch.float32)
        return X, y

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
        X_train, y_train = self._prepare_split(
            train_representations, train_metadata, "train"
        )
        X_test, y_test = self._prepare_split(
            test_representations, test_metadata, "test"
        )

        start_time = time.perf_counter()
        self.regressor.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = self.regressor.evaluate(X_test, y_test)
        train_metrics = self.regressor.evaluate(X_train, y_train)
        value = float(metrics[self.metric])

        return {
            "task": self.name,
            "regressor": self.regressor.__class__.__name__,
            "value": value,
            "normalized_value": self.normalize_metric(value, self.m_min, self.m_max),
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson_r"],
            "train_mse": train_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "training_time": training_time,
        }

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
        test_metadata = (
            self.extract_metadata(testset) if testset is not None else train_metadata
        )

        train_targets = [
            t
            for idx, meta in enumerate(train_metadata)
            for t in self._parse_clip_targets(meta, idx)
        ]
        test_targets = [
            t
            for idx, meta in enumerate(test_metadata)
            for t in self._parse_clip_targets(meta, idx)
        ]

        train_targets_np = np.array(train_targets, dtype=np.float32)
        test_targets_np = np.array(test_targets, dtype=np.float32)
        all_targets = (
            np.concatenate([train_targets_np, test_targets_np])
            if testset is not None
            else train_targets_np
        )

        frame_len_sample = len(
            parse_frame_targets(
                train_metadata[0][self.target_key],
                key_name=self.target_key,
            )
        )

        train_mean = float(np.mean(train_targets_np))
        y_pred_baseline = np.full_like(test_targets_np, train_mean)

        baseline_mse = float(mean_squared_error(test_targets_np, y_pred_baseline))
        try:
            baseline_r2 = float(r2_score(test_targets_np, y_pred_baseline))
        except (ValueError, ZeroDivisionError):
            baseline_r2 = 0.0

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
            "baseline_value": baseline_r2,
            "baseline_normalized_value": self.normalize_metric(
                baseline_r2, self.m_min, self.m_max
            ),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": float(np.sqrt(baseline_mse)),
            "baseline_mae": float(
                mean_absolute_error(test_targets_np, y_pred_baseline)
            ),
        }
