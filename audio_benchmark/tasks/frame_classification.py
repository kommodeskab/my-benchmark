"""Frame-level audio classification task."""

import time
from collections import Counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from audio_benchmark.classifiers.models import BaseClassifier
from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.downsampling import (
    align_features_to_targets,
    parse_frame_targets,
)


class FrameClassificationTask(BaseTask):
    """Frame-level audio classification task.

    Evaluates frame-level audio classification by linearly interpolating representation
    features to match the sequence length of raw ground-truth targets.
    Raw ground-truth targets are never modified.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        name: str = "frame_classification",
        label_key: str = "frame_labels",
        ignore_label: Any | None = None,
    ):
        """Initialize the FrameClassificationTask.

        Args:
            classifier: Downstream classifier instance conforming to BaseClassifier.
            name: Identifier for the task.
            label_key: Metadata key containing per-frame labels (flat torch.Tensor or list).
            ignore_label: Optional label value to exclude from training and evaluation
                (e.g. 0 for unvoiced/silence frames). Defaults to None.

        Raises:
            TypeError: If classifier is not an instance of BaseClassifier.
        """
        super().__init__(name=name)
        if not isinstance(classifier, BaseClassifier):
            raise TypeError(
                f"classifier must be an instantiated BaseClassifier instance, got {type(classifier).__name__}."
            )

        self.classifier = classifier
        self.label_key = label_key
        self.ignore_label = ignore_label
        self.metric = "accuracy"
        self.m_min = 0.0
        self.m_max = 1.0

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains valid per-frame labels.

        Args:
            dataset: PyTorch dataset to validate.
        """
        validate_dataset(dataset, required_metadata_keys=[self.label_key])
        parsed = parse_dataset_item(dataset[0])
        parse_frame_targets(parsed["metadata"][self.label_key], key_name=self.label_key)

    def _prepare_split(
        self,
        representations: Any,
        metadata: list[dict[str, Any]],
        split_name: str,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Align representations to frame targets and filter ignored labels for a split."""
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} {split_name} representations vs "
                f"{len(metadata)} {split_name} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError(f"No {split_name} representations provided for evaluation.")

        aligned_list: list[torch.Tensor] = []
        flat_labels: list[Any] = []
        for idx, meta in enumerate(metadata):
            if self.label_key not in meta:
                raise InvalidDatasetError(
                    f"{split_name.capitalize()} sample at index {idx} metadata does not contain '{self.label_key}'."
                )
            raw_labels = parse_frame_targets(
                meta[self.label_key], key_name=self.label_key
            )
            clip_reps = representations[idx]
            if "rep_length" in meta:
                clip_reps = clip_reps[: int(meta["rep_length"])]
            aligned_list.append(
                align_features_to_targets(
                    representations=clip_reps,
                    target_len=len(raw_labels),
                )
            )
            flat_labels.extend(raw_labels)

        X = torch.cat(aligned_list, dim=0)
        if self.ignore_label is not None:
            mask = [lbl != self.ignore_label for lbl in flat_labels]
            X = X[torch.tensor(mask, dtype=torch.bool)]
            flat_labels = [
                lbl for lbl, keep in zip(flat_labels, mask, strict=False) if keep
            ]
            if not flat_labels:
                raise TaskError(
                    f"No valid frames remaining after filtering ignore_label={self.ignore_label!r}."
                )

        return X, flat_labels

    def _extract_filtered_labels(self, metadata: list[dict[str, Any]]) -> list[Any]:
        """Extract flat frame labels from metadata, applying ignore_label if set."""
        labels: list[Any] = []
        for meta in metadata:
            labels.extend(
                parse_frame_targets(meta[self.label_key], key_name=self.label_key)
            )
        if self.ignore_label is not None:
            labels = [lbl for lbl in labels if lbl != self.ignore_label]
            if not labels:
                raise TaskError(
                    f"No valid frames remaining after filtering ignore_label={self.ignore_label!r}."
                )
        return labels

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Align representation features to raw frame targets and evaluate classifier.

        Ground-truth targets are never downsampled or modified. Features are aligned
        to targets via 1D linear interpolation.

        Args:
            train_representations: Tensor of shape (N_train, T', C') or sequence of representations.
            train_metadata: List of metadata dicts corresponding to each train clip.
            test_representations: Tensor of shape (N_test, T', C') or sequence of representations.
            test_metadata: List of metadata dicts corresponding to each test clip.

        Returns:
            dict[str, Any]: Frame-level classification metrics.
        """
        X_train, train_flat = self._prepare_split(
            train_representations, train_metadata, "train"
        )
        X_test, test_flat = self._prepare_split(
            test_representations, test_metadata, "test"
        )

        encoder = LabelEncoder()
        encoder.fit(train_flat + test_flat)
        num_classes = len(encoder.classes_)
        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        y_train = torch.as_tensor(encoder.transform(train_flat), dtype=torch.long)
        y_test = torch.as_tensor(encoder.transform(test_flat), dtype=torch.long)

        start_time = time.perf_counter()
        self.classifier.fit(X_train, y_train)
        training_time = float(time.perf_counter() - start_time)

        metrics = self.classifier.evaluate(X_test, y_test)
        train_metrics = self.classifier.evaluate(X_train, y_train)
        value = float(metrics[self.metric])

        return {
            "task": self.name,
            "classifier": self.classifier.__class__.__name__,
            "value": value,
            "normalized_value": self.normalize_metric(value, self.m_min, self.m_max),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "training_time": training_time,
        }

    def examine_data(
        self,
        dataset: Dataset,
        testset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Examine the dataset in the context of frame classification.

        Validates the datasets, computes clip/frame distributions, class counts,
        and evaluates a trivial baseline (majority frame class) on raw test targets.

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

        train_labels = self._extract_filtered_labels(train_metadata)
        test_labels = self._extract_filtered_labels(test_metadata)
        all_labels = train_labels + (test_labels if testset is not None else [])

        encoder = LabelEncoder()
        encoder.fit(all_labels)
        classes = [str(c) for c in encoder.classes_]
        num_classes = len(classes)
        if num_classes < 2:
            raise TaskError(
                f"Frame classification requires at least 2 distinct classes, but found {num_classes}."
            )

        frame_len_sample = len(
            parse_frame_targets(
                train_metadata[0][self.label_key], key_name=self.label_key
            )
        )
        train_counts = Counter(train_labels)
        class_counts = {str(c): int(cnt) for c, cnt in train_counts.items()}
        majority_raw = train_counts.most_common(1)[0][0]

        y_test_encoded = encoder.transform(test_labels)
        y_pred_baseline = np.full_like(
            y_test_encoded, encoder.transform([majority_raw])[0]
        )

        baseline_accuracy = float(accuracy_score(y_test_encoded, y_pred_baseline))
        baseline_macro_f1 = float(
            f1_score(y_test_encoded, y_pred_baseline, average="macro", zero_division=0)
        )
        baseline_micro_f1 = float(
            f1_score(y_test_encoded, y_pred_baseline, average="micro", zero_division=0)
        )

        return {
            "task": self.name,
            "total_clips": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_clips": len(train_metadata),
            "test_clips": len(test_metadata) if testset is not None else 0,
            "frame_length": int(frame_len_sample),
            "total_frames": len(train_labels)
            + (len(test_labels) if testset is not None else 0),
            "train_frames": len(train_labels),
            "test_frames": len(test_labels) if testset is not None else 0,
            "num_classes": int(num_classes),
            "classes": classes,
            "class_counts": class_counts,
            "majority_class": str(majority_raw),
            "baseline_strategy": "majority_class",
            "baseline_metric": self.metric,
            "baseline_value": baseline_accuracy,
            "baseline_normalized_value": self.normalize_metric(
                baseline_accuracy, self.m_min, self.m_max
            ),
            "baseline_accuracy": baseline_accuracy,
            "baseline_macro_f1": baseline_macro_f1,
            "baseline_micro_f1": baseline_micro_f1,
            "chance_level": float(1.0 / num_classes),
        }
