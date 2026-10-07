"""Variable-length sequence modeling task with pluggable sequence probes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.sequence_models import (
    BaseSequenceProbe,
)
from audio_benchmark.tasks.sequence_utils import Vocabulary, compute_token_error_rate


class SequenceModelingTask(BaseTask):
    """Task for modeling variable-length target token sequences (e.g. phonemes or words).

    Accepts any downstream probe implementing `BaseSequenceProbe` (such as `CTCSequenceProbe`
    or `TransformerDecoderProbe`).

    Args:
        probe: Explicit instantiated probe model.
        name: Identifier for the task.
        target_key: Key in sample metadata containing the target token sequence.
        metric: Primary metric for the task ('token_error_rate', 'sequence_accuracy').
    """

    def __init__(
        self,
        probe: BaseSequenceProbe,
        name: str = "sequence_modeling",
        target_key: str = "tokens",
        metric: str = "token_error_rate",
    ):
        """Initialize the SequenceModelingTask.

        Args:
            probe: Explicit instantiated BaseSequenceProbe instance.
            name: Identifier for the task.
            target_key: Key in sample metadata containing the target token sequence.
            metric: Primary metric for the task ('token_error_rate', 'sequence_accuracy').
        """
        super().__init__(name=name)
        if probe is None:
            raise TypeError(
                "Passing None as a sequence probe is not allowed. "
                "Please provide an explicit BaseSequenceProbe (e.g. CTCSequenceProbe() or TransformerDecoderProbe())."
            )
        if not isinstance(probe, BaseSequenceProbe):
            raise TypeError(
                f"probe must be an instance of BaseSequenceProbe, got {type(probe).__name__}. "
                f"Passing strings as models is not supported."
            )
        valid_metrics = ("token_error_rate", "sequence_accuracy")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {list(valid_metrics)}"
            )
        self.probe = probe
        self.target_key = target_key
        self.metric = metric

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains non-empty target sequences.

        Args:
            dataset: PyTorch dataset to validate.
        """
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        parsed = parse_dataset_item(dataset[0])
        seq = parsed["metadata"][self.target_key]
        if not hasattr(seq, "__len__"):
            raise InvalidDatasetError(
                f"Metadata key '{self.target_key}' must be a sequence (e.g. list of tokens), "
                f"got {type(seq).__name__}."
            )
        if len(seq) == 0:
            raise InvalidDatasetError(
                f"Sample at index 0 has empty sequence for key '{self.target_key}'."
            )

    def evaluate(
        self,
        train_representations: torch.Tensor,
        train_metadata: list[dict[str, Any]],
        test_representations: torch.Tensor,
        test_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train probe on train representations and evaluate on test representations.

        Args:
            train_representations: Tensor of shape (N_train, T', C').
            train_metadata: List of metadata dicts for train samples.
            test_representations: Tensor of shape (N_test, T', C').
            test_metadata: List of metadata dicts for test samples.

        Returns:
            dict[str, Any]: Evaluation metrics dictionary.
        """
        raw_train_targets = []
        for idx, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Train sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = meta[self.target_key]
            if not isinstance(seq, (list, tuple)):
                raise InvalidDatasetError(
                    f"Target for sample {idx} must be a list or tuple of tokens, got {type(seq).__name__}."
                )
            raw_train_targets.append(list(seq))

        raw_test_targets = []
        for idx, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Test sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = meta[self.target_key]
            if not isinstance(seq, (list, tuple)):
                raise InvalidDatasetError(
                    f"Target for test sample {idx} must be a list or tuple of tokens, got {type(seq).__name__}."
                )
            raw_test_targets.append(list(seq))

        vocab = Vocabulary(raw_train_targets + raw_test_targets)
        vocab_size = len(vocab)
        if vocab_size < 1:
            raise TaskError("Target vocabulary is empty.")

        encoded_train_targets = [vocab.encode(seq) for seq in raw_train_targets]

        train_reps = train_representations.to(torch.float32)
        test_reps = test_representations.to(torch.float32)

        start_time = time.perf_counter()
        self.probe.fit(train_reps, encoded_train_targets, vocab_size=vocab_size)
        training_time = float(time.perf_counter() - start_time)

        pred_id_sequences = self.probe.predict(test_reps)
        pred_tokens = [vocab.decode(ids) for ids in pred_id_sequences]

        ter, seq_acc = compute_token_error_rate(raw_test_targets, pred_tokens)
        main_value = ter if self.metric == "token_error_rate" else seq_acc

        return {
            "task": self.name,
            "probe": type(self.probe).__name__,
            "token_error_rate": ter,
            "sequence_accuracy": seq_acc,
            "value": main_value,
            "main_metric": self.metric,
            "training_time": training_time,
        }

    def examine_data(self, dataset: Any, testset: Any | None = None) -> dict[str, Any]:
        """Examine the dataset in the context of sequence modeling.

        Args:
            dataset: Audio dataset to examine for training.
            testset: Optional audio dataset to examine for testing.

        Returns:
            dict[str, Any]: Dictionary containing dataset metadata and baseline metrics.
        """
        self._validate_for_examination(dataset, [self.target_key])
        train_metadata = self.extract_metadata(dataset)

        if testset is not None:
            self._validate_for_examination(testset, [self.target_key])
            test_metadata = self.extract_metadata(testset)
        else:
            test_metadata = train_metadata

        if len(train_metadata) == 0:
            raise TaskError("Cannot examine an empty dataset.")

        raw_train_targets = []
        seq_lengths = []
        for idx, meta in enumerate(train_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = list(meta[self.target_key])
            raw_train_targets.append(seq)
            seq_lengths.append(len(seq))

        raw_test_targets = []
        for idx, meta in enumerate(test_metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = list(meta[self.target_key])
            raw_test_targets.append(seq)
            seq_lengths.append(len(seq))

        vocab = Vocabulary(raw_train_targets + raw_test_targets)
        vocab_size = len(vocab)
        if vocab_size < 1:
            raise TaskError("Target vocabulary is empty.")

        pred_empty = [[] for _ in raw_test_targets]
        baseline_ter, baseline_seq_acc = compute_token_error_rate(
            raw_test_targets, pred_empty
        )
        baseline_value = (
            baseline_ter if self.metric == "token_error_rate" else baseline_seq_acc
        )

        return {
            "task": self.name,
            "total_clips": len(train_metadata)
            + (len(test_metadata) if testset is not None else 0),
            "train_clips": len(train_metadata),
            "test_clips": len(test_metadata) if testset is not None else 0,
            "vocab_size": int(vocab_size),
            "vocabulary": list(vocab.tokens),
            "target_key": self.target_key,
            "min_sequence_length": min(seq_lengths),
            "max_sequence_length": max(seq_lengths),
            "mean_sequence_length": float(np.mean(seq_lengths)),
            "baseline_strategy": "empty_sequence",
            "baseline_metric": self.metric,
            "baseline_value": baseline_value,
            "baseline_token_error_rate": baseline_ter,
            "baseline_sequence_accuracy": baseline_seq_acc,
        }
