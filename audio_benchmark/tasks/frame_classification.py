"""Frame-level audio classification task."""

import time
from collections import Counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from audio_benchmark.classifiers.models import BaseClassifier, create_classifier
from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.downsampling import downsample_sequence


class FrameClassificationTask(BaseTask):
    """Frame-level audio classification task.

    Each frame of the input audio has a ground truth label in sample metadata.
    When representation models downsample audio along the temporal dimension
    from T to T', this task downsamples the frame labels to T' using the
    most common label in each downsampled region.
    """

    def __init__(
        self,
        name: str = "frame_classification",
        label_key: str = "frame_labels",
        classifier: str | BaseClassifier = "knn",
        test_size: float = 0.2,
        random_state: int = 42,
        split_key: str | None = None,
        classifier_kwargs: dict[str, Any] | None = None,
        metric: str = "accuracy",
    ):
        """Initialize the FrameClassificationTask.

        Args:
            name: Identifier for the task.
            label_key: Metadata key containing the per-frame label sequence (length T).
            classifier: Downstream classifier ('linear', 'knn', 'mlp', or BaseClassifier instance).
            test_size: Proportion of audio clips for testing.
            random_state: Random seed for train/test split.
            split_key: Optional metadata key ('train'/'test') for pre-defined splits.
            classifier_kwargs: Kwargs passed to the classifier.
            metric: Primary metric for the task ('accuracy', 'macro_f1', 'micro_f1').
        """
        super().__init__(name=name)
        valid_metrics = ("accuracy", "macro_f1", "micro_f1")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        self.metric = metric
        self.label_key = label_key
        self.classifier_arg = classifier
        self.test_size = test_size
        self.random_state = random_state
        self.split_key = split_key
        self.classifier_kwargs = classifier_kwargs or {}

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains per-frame labels matching audio length T."""
        validate_dataset(dataset, required_metadata_keys=[self.label_key])

        # Validate that the frame labels match audio length
        parsed = parse_dataset_item(dataset[0])
        audio_length = parsed["waveform"].shape[-1]
        labels = parsed["metadata"][self.label_key]
        if not hasattr(labels, "__len__"):
            raise InvalidDatasetError(
                f"Metadata key '{self.label_key}' must be a sequence of length {audio_length}, "
                f"got {type(labels).__name__}."
            )
        if len(labels) != audio_length:
            raise InvalidDatasetError(
                f"Length mismatch in '{self.label_key}': audio length is {audio_length} frames, "
                f"but metadata '{self.label_key}' has {len(labels)} elements."
            )

    def evaluate(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Downsample frame labels to T' and evaluate downstream classifier.

        Args:
            representations: Tensor of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each clip.

        Returns:
            dict[str, Any]: Frame-level classification metrics.
        """
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} representations vs {len(metadata)} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError("No representations provided for evaluation.")

        num_clips, target_len, feature_dim = representations.shape

        # Downsample labels for each clip from T to T' using mode (most common label)
        downsampled_labels_per_clip: list[np.ndarray] = []

        for idx, meta in enumerate(metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            raw_labels = meta[self.label_key]
            downsampled = downsample_sequence(
                raw_labels, target_len=target_len, mode="mode"
            )
            downsampled_labels_per_clip.append(downsampled)

        # Encode labels globally
        all_labels_flat = np.concatenate(downsampled_labels_per_clip)
        encoder = LabelEncoder()
        all_labels_encoded = encoder.fit_transform(all_labels_flat)
        encoded_per_clip = all_labels_encoded.reshape(num_clips, target_len)
        num_classes = len(encoder.classes_)

        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        features_np = representations.detach().cpu().numpy()  # (N, T', C')

        # Split at the clip level to avoid temporal leakage between frames of the same clip
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

        # Flatten frames for training and testing: (N_subset * T', C')
        X_train = features_np[train_clips].reshape(-1, feature_dim)
        y_train = encoded_per_clip[train_clips].reshape(-1)

        X_test = features_np[test_clips].reshape(-1, feature_dim)
        y_test = encoded_per_clip[test_clips].reshape(-1)

        # Instantiate classifier
        classifier_inst = create_classifier(
            self.classifier_arg, **self.classifier_kwargs
        )
        classifier_name = (
            self.classifier_arg
            if isinstance(self.classifier_arg, str)
            else classifier_inst.__class__.__name__
        )

        # Fit classifier with timing
        start_time = time.perf_counter()
        classifier_inst.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        # Evaluate on test frames
        metrics = classifier_inst.evaluate(X_test, y_test)
        train_metrics = classifier_inst.evaluate(X_train, y_train)

        return {
            "task": self.name,
            "classifier": str(classifier_name).lower(),
            "value": float(metrics[self.metric]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "training_time": training_time,
        }

    def examine_data(self, dataset: Dataset) -> dict[str, Any]:
        """Examine the dataset in the context of frame classification.

        Validates the dataset, computes clip/frame distributions, class counts,
        and evaluates a trivial baseline (majority frame class) on test frames.

        Args:
            dataset: Audio dataset to examine.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.label_key])
        metadata = self.extract_metadata(dataset)

        if len(metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        total_clips = len(metadata)
        frame_lengths = []
        all_labels = []

        for idx, meta in enumerate(metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            labels = meta[self.label_key]
            frame_lengths.append(len(labels))
            all_labels.extend(labels)

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)

        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

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
        total_frames = len(all_labels)
        frame_len_sample = frame_lengths[0] if frame_lengths else 0

        # Class counts across all frames
        class_counts = {str(c): int(count) for c, count in Counter(all_labels).items()}

        # Majority class in training clips
        train_labels = [
            label for i in train_clips for label in metadata[i][self.label_key]
        ]
        test_labels = [
            label for i in test_clips for label in metadata[i][self.label_key]
        ]

        train_counts = Counter(train_labels)
        majority_raw = train_counts.most_common(1)[0][0]
        majority_class = str(majority_raw)

        y_test_encoded = encoder.transform(test_labels)
        majority_encoded = encoder.transform([majority_raw])[0]
        y_pred_baseline = np.full_like(y_test_encoded, majority_encoded)

        baseline_acc = float(accuracy_score(y_test_encoded, y_pred_baseline))
        baseline_macro_f1 = float(
            f1_score(y_test_encoded, y_pred_baseline, average="macro", zero_division=0)
        )
        baseline_micro_f1 = float(
            f1_score(y_test_encoded, y_pred_baseline, average="micro", zero_division=0)
        )

        if self.metric == "accuracy":
            baseline_value = baseline_acc
        elif self.metric == "macro_f1":
            baseline_value = baseline_macro_f1
        elif self.metric == "micro_f1":
            baseline_value = baseline_micro_f1
        else:
            baseline_value = baseline_acc

        return {
            "task": self.name,
            "total_clips": int(total_clips),
            "train_clips": len(train_clips),
            "test_clips": len(test_clips),
            "frame_length": int(frame_len_sample),
            "total_frames": int(total_frames),
            "train_frames": int(train_frames_count),
            "test_frames": int(test_frames_count),
            "num_classes": int(num_classes),
            "classes": classes,
            "class_counts": class_counts,
            "majority_class": majority_class,
            "baseline_strategy": "majority_class",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_accuracy": baseline_acc,
            "baseline_macro_f1": baseline_macro_f1,
            "baseline_micro_f1": baseline_micro_f1,
            "chance_level": float(1.0 / num_classes) if num_classes > 0 else 0.0,
        }
