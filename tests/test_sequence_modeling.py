"""Tests for SequenceModelingTask, CTC probe, Transformer Decoder probe, and sequence utilities."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    SequenceModelingTask,
    compute_token_error_rate,
    levenshtein_distance,
)
from audio_benchmark.exceptions import InvalidDatasetError
from audio_benchmark.tasks.sequence_utils import Vocabulary

# ==============================================================================
# 1. Sequence Utility Tests
# ==============================================================================


def test_levenshtein_distance():
    # Identity
    assert levenshtein_distance([], []) == 0
    assert levenshtein_distance(["a", "b"], ["a", "b"]) == 0

    # Insertions & deletions
    assert levenshtein_distance([], ["a", "b"]) == 2
    assert levenshtein_distance(["a", "b"], []) == 2
    assert levenshtein_distance(["a"], ["a", "b"]) == 1

    # Substitutions
    assert levenshtein_distance(["a", "b"], ["a", "c"]) == 1

    # Classic kitten -> sitting (3 edits: k->s, e->i, +g)
    assert levenshtein_distance(list("kitten"), list("sitting")) == 3


def test_compute_token_error_rate():
    refs = [["a", "b", "c"], ["d", "e"]]
    hyps = [["a", "b", "c"], ["d", "e"]]
    ter, acc = compute_token_error_rate(refs, hyps)
    assert ter == 0.0
    assert acc == 1.0

    # 1 error in 5 tokens total (edit distance 1 on second sequence)
    hyps_with_error = [["a", "b", "c"], ["d", "f"]]
    ter, acc = compute_token_error_rate(refs, hyps_with_error)
    assert pytest.approx(ter) == 1.0 / 5.0
    assert acc == 0.5


def test_vocabulary_encoding_decoding():
    sequences = [["ph_0", "ph_1"], ["ph_1", "ph_2"]]
    vocab = Vocabulary(sequences)

    assert len(vocab) == 3
    encoded = vocab.encode(["ph_0", "ph_2"])
    assert len(encoded) == 2
    decoded = vocab.decode(encoded)
    assert decoded == ["ph_0", "ph_2"]


# ==============================================================================
# 2. SequenceModelingTask Validation Tests
# ==============================================================================


class MissingKeyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, 100),
            "sample_rate": 16000,
            "metadata": {"id": idx},
        }


class NonSequenceDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, 100),
            "sample_rate": 16000,
            "metadata": {"tokens": 123},
        }


class EmptySequenceDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(1, 100),
            "sample_rate": 16000,
            "metadata": {"tokens": []},
        }


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown sequence modeling method"):
        SequenceModelingTask(method="rnn")


def test_dataset_validation_missing_key():
    task = SequenceModelingTask(target_key="tokens")
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        task.validate_dataset(MissingKeyDataset())


def test_dataset_validation_non_sequence():
    task = SequenceModelingTask(target_key="tokens")
    with pytest.raises(InvalidDatasetError, match="must be a sequence"):
        task.validate_dataset(NonSequenceDataset())


def test_dataset_validation_empty_sequence():
    task = SequenceModelingTask(target_key="tokens")
    with pytest.raises(InvalidDatasetError, match="empty sequence"):
        task.validate_dataset(EmptySequenceDataset())


# ==============================================================================
# 3. Model Training & Evaluation Tests
# ==============================================================================


def test_sequence_modeling_ctc():
    task = SequenceModelingTask(
        method="ctc",
        test_size=0.25,
        random_state=42,
        epochs=1,
        batch_size=4,
        device="cpu",
    )

    # 8 clips, representations (B=8, T'=10, C'=16)
    reps = torch.randn(8, 10, 16)
    meta = [
        {"tokens": [f"tok_{(i + k) % 3}" for k in range(2 + (i % 2))]} for i in range(8)
    ]

    # Test examine_data
    info = task.examine_data(meta)
    assert info["train_clips"] == 6
    assert info["test_clips"] == 2
    assert info["vocab_size"] == 3
    assert info["baseline_strategy"] == "empty_sequence"
    assert info["baseline_metric"] == "token_error_rate"
    assert info["baseline_value"] == 1.0
    assert info["baseline_token_error_rate"] == 1.0
    assert info["baseline_sequence_accuracy"] == 0.0

    res = task.evaluate(reps, meta)
    assert res["task"] == "sequence_modeling"
    assert res["method"] == "ctc"
    assert "train_clips" not in res
    assert "test_clips" not in res
    assert "vocab_size" not in res
    assert 0.0 <= res["sequence_accuracy"] <= 1.0
    assert res["token_error_rate"] >= 0.0
    assert "training_time" in res
    assert res["training_time"] >= 0.0


def test_sequence_modeling_transformer():
    task = SequenceModelingTask(
        method="transformer",
        test_size=0.25,
        random_state=42,
        epochs=1,
        batch_size=4,
        device="cpu",
        model_kwargs={
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
        },
    )

    reps = torch.randn(8, 10, 16)
    meta = [
        {"tokens": [f"tok_{(i + k) % 3}" for k in range(2 + (i % 2))]} for i in range(8)
    ]

    res = task.evaluate(reps, meta)
    assert res["task"] == "sequence_modeling"
    assert res["method"] == "transformer"
    assert "train_clips" not in res
    assert "test_clips" not in res
    assert "vocab_size" not in res
    assert 0.0 <= res["sequence_accuracy"] <= 1.0
    assert res["token_error_rate"] >= 0.0
    assert "training_time" in res
    assert res["training_time"] >= 0.0


def test_sequence_modeling_split_key():
    task = SequenceModelingTask(method="ctc", split_key="split", epochs=1, device="cpu")
    reps = torch.randn(6, 10, 16)
    meta = [
        {"tokens": ["a", "b"], "split": "train" if i < 4 else "test"} for i in range(6)
    ]

    info = task.examine_data(meta)
    assert info["train_clips"] == 4
    assert info["test_clips"] == 2

    res = task.evaluate(reps, meta)
    assert "train_clips" not in res
    assert "test_clips" not in res


# ==============================================================================
# 4. Benchmark Suite Integration Tests
# ==============================================================================


def test_benchmark_runner_with_sequence_tasks(tmp_path, capsys):
    benchmark = AudioBenchmark(cache_dir=tmp_path, enable_cache=True)
    dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # CTC method
    task_ctc = SequenceModelingTask(method="ctc", epochs=1, batch_size=4, device="cpu")
    res_ctc = benchmark.run(
        model=model,
        model_name="dummy_seq_model",
        dataset=dataset,
        task=task_ctc,
        device="cpu",
    )
    assert res_ctc.task_name == "sequence_modeling"
    assert "token_error_rate" in res_ctc.metrics
    assert "sequence_accuracy" in res_ctc.metrics

    # Transformer method
    task_tf = SequenceModelingTask(
        method="transformer",
        epochs=1,
        batch_size=4,
        device="cpu",
        model_kwargs={
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
        },
    )
    res_tf = benchmark.run(
        model=model,
        model_name="dummy_seq_model",
        dataset=dataset,
        task=task_tf,
        device="cpu",
    )
    assert res_tf.task_name == "sequence_modeling"
    assert "token_error_rate" in res_tf.metrics
    assert "sequence_accuracy" in res_tf.metrics

    # BenchmarkResult print_summary test
    res_ctc.print_summary()
    res_tf.print_summary()
    captured = capsys.readouterr()
    assert "Token Error Rate:" in captured.out
    assert "Sequence Acc:" in captured.out
    assert "Vocab Size:" in captured.out
