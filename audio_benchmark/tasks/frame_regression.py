"""Frame-level audio regression task."""

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.regressors.models import BaseRegressor, create_regressor
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.downsampling import downsample_sequence


class FrameRegressionTask(BaseTask):
    """Frame-level audio regression task.

    Each frame of the input audio has a continuous target in sample metadata.
    When representation models downsample audio along the temporal dimension
    from T to T', this task downsamples the frame targets to T' by computing
    the mean of values in each downsampled region.
    """

    def __init__(
        self,
        name: str = "frame_regression",
        target_key: str = "frame_targets",
        regressor: str | BaseRegressor = "linear",
        test_size: float = 0.2,
        random_state: int = 42,
        split_key: str | None = None,
        regressor_kwargs: dict[str, Any] | None = None,
        metric: str = "r2",
    ):
        """Initialize the FrameRegressionTask.

        Args:
            name: Identifier for the task.
            target_key: Metadata key containing the per-frame target sequence (length T).
            regressor: Downstream regressor ('linear', 'knn', or 'mlp').
            test_size: Proportion of audio clips for testing.
            random_state: Random seed for train/test split.
            split_key: Optional metadata key ('train'/'test') for pre-defined splits.
            regressor_kwargs: Kwargs passed to the regressor.
            metric: Primary metric for the task ('r2', 'mse', 'rmse', 'mae', 'pearson_r').
        """
        super().__init__(name=name)
        valid_metrics = ("r2", "mse", "rmse", "mae", "pearson_r")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        self.metric = metric
        self.target_key = target_key
        self.regressor_arg = regressor
        self.test_size = test_size
        self.random_state = random_state
        self.split_key = split_key
        self.regressor_kwargs = regressor_kwargs or {}

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains per-frame targets matching audio length T."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        # Validate that the frame targets match audio length
        parsed = parse_dataset_item(dataset[0])
        audio_length = parsed["waveform"].shape[-1]
        targets = parsed["metadata"][self.target_key]
        if not hasattr(targets, "__len__"):
            raise InvalidDatasetError(
                f"Metadata key '{self.target_key}' must be a sequence of length {audio_length}, "
                f"got {type(targets).__name__}."
            )
        if len(targets) != audio_length:
            raise InvalidDatasetError(
                f"Length mismatch in '{self.target_key}': audio length is {audio_length} frames, "
                f"but metadata '{self.target_key}' has {len(targets)} elements."
            )

    def evaluate(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Downsample frame targets to T' and evaluate downstream regressor.

        Args:
            representations: Tensor of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each clip.

        Returns:
            dict[str, Any]: Frame-level regression metrics.
        """
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} representations vs {len(metadata)} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError("No representations provided for evaluation.")

        num_clips, target_len, feature_dim = representations.shape

        # Downsample continuous targets for each clip from T to T' using mean
        downsampled_targets_per_clip: list[np.ndarray] = []

        for idx, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            raw_targets = meta[self.target_key]
            downsampled = downsample_sequence(
                raw_targets, target_len=target_len, mode="mean"
            )
            downsampled_targets_per_clip.append(downsampled)

        targets_per_clip = np.array(
            downsampled_targets_per_clip, dtype=np.float32
        )  # (N, T')
        features_np = representations.detach().cpu().numpy()  # (N, T', C')

        # Split at the clip level to avoid temporal leakage
        clip_indices = np.arange(num_clips)
        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_clips = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "train"
            ]
            test_clips = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "test"
            ]
            if len(train_clips) == 0 or len(test_clips) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(train_clips)} train clips, {len(test_clips)} test clips."
                )
        else:
            train_clips, test_clips = train_test_split(
                clip_indices, test_size=self.test_size, random_state=self.random_state
            )

        # Flatten frames for training and testing
        X_train = features_np[train_clips].reshape(-1, feature_dim)
        y_train = targets_per_clip[train_clips].reshape(-1)

        X_test = features_np[test_clips].reshape(-1, feature_dim)
        y_test = targets_per_clip[test_clips].reshape(-1)

        # Fit regressor
        regressor_inst = create_regressor(self.regressor_arg, **self.regressor_kwargs)
        regressor_name = (
            self.regressor_arg
            if isinstance(self.regressor_arg, str)
            else regressor_inst.__class__.__name__
        )
        # Fit regressor with timing
        start_time = time.perf_counter()
        regressor_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        # Evaluate on test frames
        metrics = regressor_inst.evaluate(X_test, y_test)
        train_metrics = regressor_inst.evaluate(X_train, y_train)

        return {
            "task": self.name,
            "regressor": str(regressor_name).lower(),
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

    def examine_data(self, dataset: Dataset) -> dict[str, Any]:
        """Examine the dataset in the context of frame regression.

        Validates the dataset, computes frame target distributions and train/test splits,
        and computes a trivial baseline (frame target mean predictor) expected metric.

        Args:
            dataset: Audio dataset to examine.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.target_key])
        metadata = self.extract_metadata(dataset)

        if len(metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        total_clips = len(metadata)
        frame_lengths = []
        all_targets = []

        for idx, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            targets = meta[self.target_key]
            frame_lengths.append(len(targets))
            try:
                all_targets.extend([float(t) for t in targets])
            except (ValueError, TypeError) as e:
                raise TaskError(
                    f"Sample at index {idx} frame target cannot be converted to float: {e}"
                ) from e

        all_targets_np = np.array(all_targets, dtype=np.float32)
        clip_indices = np.arange(total_clips)

        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_clips = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "train"
            ]
            test_clips = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "test"
            ]
            if len(train_clips) == 0 or len(test_clips) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(train_clips)} train clips, {len(test_clips)} test clips."
                )
        else:
            train_clips, test_clips = train_test_split(
                clip_indices, test_size=self.test_size, random_state=self.random_state
            )

        train_frames_count = sum(frame_lengths[i] for i in train_clips)
        test_frames_count = sum(frame_lengths[i] for i in test_clips)
        total_frames = len(all_targets)
        frame_len_sample = frame_lengths[0] if frame_lengths else 0

        train_targets = np.array(
            [float(t) for i in train_clips for t in metadata[i][self.target_key]],
            dtype=np.float32,
        )
        test_targets = np.array(
            [float(t) for i in test_clips for t in metadata[i][self.target_key]],
            dtype=np.float32,
        )

        train_mean = float(np.mean(train_targets))
        y_pred_baseline = np.full_like(test_targets, train_mean)

        baseline_mse = float(mean_squared_error(test_targets, y_pred_baseline))
        baseline_rmse = float(np.sqrt(baseline_mse))
        baseline_mae = float(mean_absolute_error(test_targets, y_pred_baseline))
        try:
            baseline_r2 = float(r2_score(test_targets, y_pred_baseline))
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
            "total_clips": int(total_clips),
            "train_clips": len(train_clips),
            "test_clips": len(test_clips),
            "frame_length": int(frame_len_sample),
            "total_frames": int(total_frames),
            "train_frames": int(train_frames_count),
            "test_frames": int(test_frames_count),
            "target_key": self.target_key,
            "target_min": float(np.min(all_targets_np)),
            "target_max": float(np.max(all_targets_np)),
            "target_mean": float(np.mean(all_targets_np)),
            "target_std": float(np.std(all_targets_np)),
            "baseline_strategy": "mean_predictor",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_r2": baseline_r2,
            "baseline_mse": baseline_mse,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        }
