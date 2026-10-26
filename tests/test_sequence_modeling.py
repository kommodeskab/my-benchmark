"""Tests for SequenceModelingTask, CTC probe, Transformer Decoder probe, and sequence utilities."""

import pytest
import torch
from torch.utils.data import Dataset

from audio_benchmark import (
    AudioBenchmark,
    CTCSequenceProbe,
    DummyAudioDataset,
    DummyAudioModel,
    SequenceModelingTask,
    TransformerDecoderProbe,
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


def test_invalid_probe_raises():
    with pytest.raises(
        TypeError, match="probe must be an instance of BaseSequenceProbe"
    ):
        SequenceModelingTask(probe="ctc")
    with pytest.raises(
        TypeError, match="Passing None as a sequence probe is not allowed"
    ):
        SequenceModelingTask(probe=None)


def test_dataset_validation_missing_key():
    task = SequenceModelingTask(probe=CTCSequenceProbe(), target_key="tokens")
    with pytest.raises(InvalidDatasetError, match="missing required key"):
        task.validate_dataset(MissingKeyDataset())


def test_dataset_validation_non_sequence():
    task = SequenceModelingTask(probe=CTCSequenceProbe(), target_key="tokens")
    with pytest.raises(InvalidDatasetError, match="must be a sequence"):
        task.validate_dataset(NonSequenceDataset())


def test_dataset_validation_empty_sequence():
    task = SequenceModelingTask(probe=CTCSequenceProbe(), target_key="tokens")
    with pytest.raises(InvalidDatasetError, match="empty sequence"):
        task.validate_dataset(EmptySequenceDataset())


# ==============================================================================
# 3. Model Training & Evaluation Tests
# ==============================================================================


def test_sequence_modeling_ctc():
    probe = CTCSequenceProbe(
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    task = SequenceModelingTask(probe=probe)

    train_reps = torch.randn(6, 10, 16)
    train_meta = [
        {"tokens": [f"tok_{(i + k) % 3}" for k in range(2 + (i % 2))]} for i in range(6)
    ]
    test_reps = torch.randn(2, 10, 16)
    test_meta = [
        {"tokens": [f"tok_{(i + k) % 3}" for k in range(2 + (i % 2))]} for i in range(2)
    ]

    # Test examine_data
    info = task.examine_data(train_meta, test_meta)
    assert info["train_clips"] == 6
    assert info["test_clips"] == 2
    assert info["vocab_size"] == 3
    assert info["baseline_strategy"] == "empty_sequence"
    assert info["baseline_metric"] == "token_error_rate"
    assert info["baseline_value"] == 1.0
    assert info["baseline_token_error_rate"] == 1.0
    assert info["baseline_sequence_accuracy"] == 0.0

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert res["task"] == "sequence_modeling"
    assert res["probe"] == "CTCSequenceProbe"
    assert "train_clips" not in res
    assert "test_clips" not in res
    assert "vocab_size" not in res
    assert 0.0 <= res["sequence_accuracy"] <= 1.0
    assert res["token_error_rate"] >= 0.0
    assert "training_time" in res
    assert res["training_time"] >= 0.0


def test_sequence_modeling_transformer():
    probe = TransformerDecoderProbe(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    task = SequenceModelingTask(probe=probe)

    train_reps = torch.randn(6, 10, 16)
    train_meta = [
        {"tokens": [f"tok_{(i + k) % 3}" for k in range(2 + (i % 2))]} for i in range(6)
    ]
    test_reps = torch.randn(2, 10, 16)
    test_meta = [
        {"tokens": [f"tok_{(i + k) % 3}" for k in range(2 + (i % 2))]} for i in range(2)
    ]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert res["task"] == "sequence_modeling"
    assert res["probe"] == "TransformerDecoderProbe"
    assert "train_clips" not in res
    assert "test_clips" not in res
    assert "vocab_size" not in res
    assert 0.0 <= res["sequence_accuracy"] <= 1.0
    assert res["token_error_rate"] >= 0.0
    assert "training_time" in res
    assert res["training_time"] >= 0.0


# ==============================================================================
# 4. Benchmark Suite Integration Tests
# ==============================================================================


def test_benchmark_runner_with_sequence_tasks(tmp_path, capsys):
    train_dataset = DummyAudioDataset(num_samples=8, audio_length=320, num_classes=2)
    test_dataset = DummyAudioDataset(num_samples=4, audio_length=320, num_classes=2)
    model = DummyAudioModel(representation_dim=16)

    # CTC probe
    probe_ctc = CTCSequenceProbe(epochs=1, batch_size=4, device="cpu")
    task_ctc = SequenceModelingTask(probe=probe_ctc)
    benchmark_ctc = AudioBenchmark(
        dataset=train_dataset,
        testset=test_dataset,
        task=task_ctc,
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=True,
    )
    res_ctc = benchmark_ctc.run(
        model=model,
        model_name="dummy_seq_model",
        device="cpu",
    )
    assert res_ctc.task_name == "sequence_modeling"
    assert "token_error_rate" in res_ctc.metrics
    assert "sequence_accuracy" in res_ctc.metrics

    # Transformer probe
    probe_tf = TransformerDecoderProbe(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    task_tf = SequenceModelingTask(probe=probe_tf)
    benchmark_tf = AudioBenchmark(
        dataset=train_dataset,
        testset=test_dataset,
        task=task_tf,
        dataset_name="dummy_audio",
        cache_dir=tmp_path,
        enable_cache=True,
    )
    res_tf = benchmark_tf.run(
        model=model,
        model_name="dummy_seq_model",
        device="cpu",
    )
    assert res_tf.task_name == "sequence_modeling"
    assert "token_error_rate" in res_tf.metrics
    assert "sequence_accuracy" in res_tf.metrics

    # BenchmarkResult print_summary test
    res_ctc.print_summary()
    res_tf.print_summary()
    captured = capsys.readouterr()
    assert "token_error_rate" in captured.out
    assert "sequence_accuracy" in captured.out


def test_sequence_probes_variable_length_representations():
    """Verify dynamic per-batch zero-padding and masking for variable-length representations."""
    train_reps = [
        torch.randn(5, 16),
        torch.randn(12, 16),
        torch.randn(8, 16),
        torch.randn(15, 16),
    ]
    targets = [[0, 1], [1, 2, 0], [2], [0, 1, 2]]
    test_reps = [torch.randn(6, 16), torch.randn(14, 16)]

    tf_probe = TransformerDecoderProbe(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=1,
        batch_size=2,
        device="cpu",
        show_progress=False,
    )
    tf_probe.fit(train_reps, targets, vocab_size=3)
    tf_preds = tf_probe.predict(test_reps)
    assert len(tf_preds) == 2

    ctc_probe = CTCSequenceProbe(
        epochs=1,
        batch_size=2,
        device="cpu",
        show_progress=False,
    )
    ctc_probe.fit(train_reps, targets, vocab_size=3)
    ctc_preds = ctc_probe.predict(test_reps)
    assert len(ctc_preds) == 2


def test_sequence_modeling_task_with_rep_length_slicing():
    """Verify SequenceModelingTask slices representations to rep_length before probe fitting."""
    probe = TransformerDecoderProbe(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=1,
        batch_size=2,
        device="cpu",
        show_progress=False,
    )
    task = SequenceModelingTask(probe=probe)

    train_reps = torch.randn(4, 20, 16)
    train_meta = [
        {"tokens": ["a", "b"], "rep_length": 6},
        {"tokens": ["b", "c"], "rep_length": 14},
        {"tokens": ["a"], "rep_length": 9},
        {"tokens": ["c", "a"], "rep_length": 20},
    ]
    test_reps = torch.randn(2, 20, 16)
    test_meta = [
        {"tokens": ["a", "b"], "rep_length": 8},
        {"tokens": ["b", "c"], "rep_length": 18},
    ]

    res = task.evaluate(train_reps, train_meta, test_reps, test_meta)
    assert res["task"] == "sequence_modeling"
    assert res["probe"] == "TransformerDecoderProbe"
    assert 0.0 <= res["sequence_accuracy"] <= 1.0
