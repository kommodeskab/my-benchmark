"""Variable-length sequence modeling task (CTC and Transformer Decoder)."""

import time
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio_benchmark.dataset import parse_dataset_item, validate_dataset
from audio_benchmark.exceptions import InvalidDatasetError, TaskError
from audio_benchmark.tasks.base import BaseTask
from audio_benchmark.tasks.sequence_models import (
    CTCSequenceProbe,
    TransformerDecoderProbe,
)
from audio_benchmark.tasks.sequence_utils import Vocabulary, compute_token_error_rate
from audio_benchmark.utils import resolve_device


class _SequencePairDataset(Dataset):
    """Dataset pairing precomputed audio representations with target token sequences."""

    def __init__(self, representations: torch.Tensor, targets: list[list[int]]):
        self.representations = representations
        self.targets = targets

    def __len__(self) -> int:
        return len(self.representations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        return self.representations[idx], self.targets[idx]


def _collate_transformer_batch(
    batch: list[tuple[torch.Tensor, list[int]]],
    sos_id: int,
    eos_id: int,
    pad_id: int,
    num_special: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch with teacher-forcing inputs, targets, and padding masks."""
    reps = torch.stack([item[0] for item in batch], dim=0)  # (B, T', C')
    targets = [item[1] for item in batch]

    max_len = max(len(t) for t in targets)
    batch_size = len(batch)

    # dec_input: <sos> + tokens
    dec_input = torch.full((batch_size, max_len + 1), pad_id, dtype=torch.long)
    # dec_target: tokens + <eos>
    dec_target = torch.full((batch_size, max_len + 1), pad_id, dtype=torch.long)
    padding_mask = torch.zeros((batch_size, max_len + 1), dtype=torch.bool)

    for i, seq in enumerate(targets):
        shifted_seq = [tok + num_special for tok in seq]
        seq_len = len(shifted_seq)

        dec_input[i, 0] = sos_id
        if seq_len > 0:
            dec_input[i, 1 : seq_len + 1] = torch.tensor(shifted_seq, dtype=torch.long)

        dec_target[i, :seq_len] = torch.tensor(shifted_seq, dtype=torch.long)
        dec_target[i, seq_len] = eos_id

        # True indicates padding positions to ignore in self-attention
        padding_mask[i, seq_len + 1 :] = True

    return reps, dec_input, dec_target, padding_mask


class SequenceModelingTask(BaseTask):
    """Task for modeling variable-length target token sequences (e.g. phonemes or words).

    Supports two modeling methods:
      1. 'ctc': Linear projection trained with Connectionist Temporal Classification (CTC) loss.
      2. 'transformer': Attention-based autoregressive Transformer decoder with cross-attention.
    """

    def __init__(
        self,
        name: str = "sequence_modeling",
        target_key: str = "tokens",
        method: str = "ctc",
        test_size: float = 0.2,
        random_state: int = 42,
        split_key: str | None = None,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        model_kwargs: dict[str, Any] | None = None,
        metric: str = "token_error_rate",
    ):
        """Initialize the SequenceModelingTask.

        Args:
            name: Identifier for the task.
            target_key: Key in sample metadata containing the target token sequence.
            method: Modeling method: 'ctc' or 'transformer'.
            test_size: Proportion of audio clips for testing.
            random_state: Random seed for train/test split.
            split_key: Optional metadata key ('train'/'test') for pre-defined splits.
            epochs: Training epochs for downstream probe.
            batch_size: Batch size for training.
            lr: Learning rate for probe optimizer.
            device: Training device ('auto', 'cpu', 'cuda', etc.).
            model_kwargs: Additional kwargs passed to the probe model.
            metric: Primary metric for the task ('token_error_rate', 'sequence_accuracy').
        """
        super().__init__(name=name)
        valid_metrics = ("token_error_rate", "sequence_accuracy")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported metrics: {valid_metrics}"
            )
        self.metric = metric
        self.target_key = target_key
        self.method = method.lower().strip()
        if self.method not in ("ctc", "transformer"):
            raise ValueError(
                f"Unknown sequence modeling method: '{method}'. Supported methods: 'ctc', 'transformer'."
            )

        self.test_size = test_size
        self.random_state = random_state
        self.split_key = split_key
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device_str = device
        self.model_kwargs = model_kwargs or {}

    def validate_dataset(self, dataset: Dataset) -> None:
        """Verify that dataset metadata contains non-empty target sequences."""
        validate_dataset(dataset, required_metadata_keys=[self.target_key])

        # Validate that the first sample has a sequence target
        parsed = parse_dataset_item(dataset[0])
        seq = parsed["metadata"][self.target_key]
        if not hasattr(seq, "__len__"):
            raise InvalidDatasetError(
                f"Metadata key '{self.target_key}' must be a sequence (e.g. list of tokens), "
                f"got {type(seq).__name__}."
            )
        if len(seq) == 0:
            raise InvalidDatasetError(
                f"Metadata key '{self.target_key}' at index 0 is an empty sequence."
            )

    def evaluate(
        self,
        representations: torch.Tensor,
        metadata: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Train sequence probe on representations and compute TER & Sequence Accuracy.

        Args:
            representations: Computed representations of shape (N, T', C').
            metadata: List of metadata dicts corresponding to each clip.

        Returns:
            dict[str, Any]: Dictionary containing TER, accuracy, and sequence metrics.
        """
        if len(representations) != len(metadata):
            raise TaskError(
                f"Length mismatch: {len(representations)} representations vs {len(metadata)} metadata entries."
            )
        if len(representations) == 0:
            raise TaskError("No representations provided for evaluation.")

        num_clips, seq_len_T, feature_dim = representations.shape

        # Extract target sequences
        raw_targets = []
        for idx, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = meta[self.target_key]
            raw_targets.append(list(seq))

        # Build vocabulary
        vocab = Vocabulary(raw_targets)
        vocab_size = len(vocab)
        if vocab_size < 1:
            raise TaskError("Target vocabulary is empty.")

        encoded_targets = [vocab.encode(seq) for seq in raw_targets]
        device = resolve_device(self.device_str)

        # Train/test split at the clip level
        clip_indices = np.arange(num_clips)
        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_indices = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "train"
            ]
            test_indices = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "test"
            ]
            if len(train_indices) == 0 or len(test_indices) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(train_indices)} train clips, {len(test_indices)} test clips."
                )
        else:
            train_indices, test_indices = train_test_split(
                clip_indices, test_size=self.test_size, random_state=self.random_state
            )

        train_reps = representations[train_indices].to(torch.float32)
        train_targets = [encoded_targets[i] for i in train_indices]

        test_reps = representations[test_indices].to(torch.float32)
        test_ref_tokens = [raw_targets[i] for i in test_indices]

        train_dataset = _SequencePairDataset(train_reps, train_targets)

        # Model instantiation and training
        train_loss = 0.0

        if self.method == "ctc":
            probe = CTCSequenceProbe(
                in_dim=feature_dim, vocab_size=vocab_size, **self.model_kwargs
            ).to(device)
            criterion = nn.CTCLoss(blank=probe.blank_id, zero_infinity=True)
            optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)

            loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=lambda b: (
                    torch.stack([item[0] for item in b]),
                    [item[1] for item in b],
                ),
            )

            probe.train()
            total_loss = 0.0
            total_batches = 0
            start_time = time.perf_counter()

            for _ in range(self.epochs):
                for batch_reps, batch_targets in loader:
                    batch_reps = batch_reps.to(device)
                    batch_size_cur = batch_reps.size(0)

                    # Shift target token IDs by 1 so index 0 is reserved for blank
                    shifted_targets = [
                        [tok + 1 for tok in seq] for seq in batch_targets
                    ]
                    flat_targets = torch.tensor(
                        [tok for seq in shifted_targets for tok in seq],
                        dtype=torch.long,
                        device=device,
                    )
                    target_lengths = torch.tensor(
                        [len(seq) for seq in shifted_targets],
                        dtype=torch.long,
                        device=device,
                    )
                    input_lengths = torch.full(
                        (batch_size_cur,), seq_len_T, dtype=torch.long, device=device
                    )

                    optimizer.zero_grad()
                    log_probs = probe(batch_reps)  # (T', B, num_classes)
                    loss = criterion(
                        log_probs, flat_targets, input_lengths, target_lengths
                    )
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item())
                    total_batches += 1

            training_time = float(time.perf_counter() - start_time)
            train_loss = total_loss / max(1, total_batches)

            # Greedy best-path decode on test set
            pred_id_sequences = probe.greedy_decode(test_reps.to(device))
            pred_tokens = [vocab.decode(ids) for ids in pred_id_sequences]

        elif self.method == "transformer":
            probe = TransformerDecoderProbe(
                in_dim=feature_dim,
                vocab_size=vocab_size,
                **self.model_kwargs,
            ).to(device)
            criterion = nn.CrossEntropyLoss(ignore_index=probe.PAD_ID)
            optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)

            collate_fn = lambda b: _collate_transformer_batch(
                b,
                sos_id=probe.SOS_ID,
                eos_id=probe.EOS_ID,
                pad_id=probe.PAD_ID,
                num_special=probe.NUM_SPECIAL,
            )
            loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

            probe.train()
            total_loss = 0.0
            total_batches = 0
            start_time = time.perf_counter()

            for _ in range(self.epochs):
                for batch_reps, dec_in, dec_tgt, padding_mask in loader:
                    batch_reps = batch_reps.to(device)
                    dec_in = dec_in.to(device)
                    dec_tgt = dec_tgt.to(device)
                    padding_mask = padding_mask.to(device)

                    seq_len_dec = dec_in.size(1)
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(
                        seq_len_dec, device=device, dtype=torch.bool
                    )

                    optimizer.zero_grad()
                    logits = probe(
                        memory=batch_reps,
                        tgt=dec_in,
                        tgt_mask=causal_mask,
                        tgt_key_padding_mask=padding_mask,
                    )  # (B, S, total_tokens)

                    loss = criterion(
                        logits.reshape(-1, probe.total_tokens), dec_tgt.reshape(-1)
                    )
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item())
                    total_batches += 1

            training_time = float(time.perf_counter() - start_time)
            train_loss = total_loss / max(1, total_batches)

            # Greedy autoregressive decode on test set
            max_gen_len = max(len(t) for t in raw_targets) + 10
            pred_id_sequences = probe.generate(
                test_reps.to(device), max_len=max_gen_len
            )
            pred_tokens = [vocab.decode(ids) for ids in pred_id_sequences]

        # Compute Token Error Rate and Sequence Match Accuracy
        ter, seq_acc = compute_token_error_rate(test_ref_tokens, pred_tokens)
        main_value = ter if self.metric == "token_error_rate" else seq_acc

        return {
            "task": self.name,
            "method": self.method,
            "value": float(main_value),
            "token_error_rate": ter,
            "sequence_accuracy": seq_acc,
            "train_loss": train_loss,
            "training_time": training_time,
        }

    def examine_data(self, dataset: Dataset) -> dict[str, Any]:
        """Examine the dataset in the context of sequence modeling.

        Validates the dataset, computes vocabulary, sequence lengths, clip splits,
        and computes a trivial baseline (empty sequence prediction) expected metric.

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
        raw_targets = []
        seq_lengths = []

        for idx, meta in enumerate(metadata):
            if self.target_key not in meta:
                raise InvalidDatasetError(
                    f"Sample at index {idx} metadata does not contain '{self.target_key}'."
                )
            seq = list(meta[self.target_key])
            raw_targets.append(seq)
            seq_lengths.append(len(seq))

        vocab = Vocabulary(raw_targets)
        vocab_size = len(vocab)
        if vocab_size < 1:
            raise TaskError("Target vocabulary is empty.")

        clip_indices = np.arange(total_clips)
        if self.split_key is not None and all(self.split_key in m for m in metadata):
            train_indices = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "train"
            ]
            test_indices = [
                i for i, m in enumerate(metadata) if m[self.split_key] == "test"
            ]
            if len(train_indices) == 0 or len(test_indices) == 0:
                raise TaskError(
                    f"Invalid train/test split via split_key '{self.split_key}': "
                    f"{len(train_indices)} train clips, {len(test_indices)} test clips."
                )
        else:
            train_indices, test_indices = train_test_split(
                clip_indices, test_size=self.test_size, random_state=self.random_state
            )

        test_ref_tokens = [raw_targets[i] for i in test_indices]
        pred_empty = [[] for _ in test_indices]
        baseline_ter, baseline_seq_acc = compute_token_error_rate(
            test_ref_tokens, pred_empty
        )
        baseline_value = (
            baseline_ter if self.metric == "token_error_rate" else baseline_seq_acc
        )

        return {
            "task": self.name,
            "total_clips": int(total_clips),
            "train_clips": len(train_indices),
            "test_clips": len(test_indices),
            "vocab_size": int(vocab_size),
            "vocabulary": list(vocab.tokens),
            "target_key": self.target_key,
            "min_sequence_length": int(min(seq_lengths)),
            "max_sequence_length": int(max(seq_lengths)),
            "mean_sequence_length": float(np.mean(seq_lengths)),
            "baseline_strategy": "empty_sequence",
            "baseline_metric": self.metric,
            "baseline_value": float(baseline_value),
            "baseline_token_error_rate": float(baseline_ter),
            "baseline_sequence_accuracy": float(baseline_seq_acc),
        }
