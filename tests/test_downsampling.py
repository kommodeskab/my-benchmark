"""Tests for temporal downsampling utility."""

import numpy as np
import pytest

from audio_benchmark.tasks.downsampling import downsample_sequence


def test_downsample_mean_regression():
    # Sequence of 8 numbers downsampled to 4 (regions of size 2)
    # [1, 3] -> 2.0, [2, 4] -> 3.0, [5, 5] -> 5.0, [10, 20] -> 15.0
    seq = [1, 3, 2, 4, 5, 5, 10, 20]
    out = downsample_sequence(seq, target_len=4, mode="mean")
    assert len(out) == 4
    assert np.allclose(out, [2.0, 3.0, 5.0, 15.0])


def test_downsample_mode_classification():
    # Sequence of 6 labels downsampled to 2 (regions of size 3)
    # ['dog', 'dog', 'cat'] -> 'dog'
    # ['cat', 'cat', 'bird'] -> 'cat'
    seq = ["dog", "dog", "cat", "cat", "cat", "bird"]
    out = downsample_sequence(seq, target_len=2, mode="mode")
    assert len(out) == 2
    assert list(out) == ["dog", "cat"]


def test_downsample_identical_length():
    seq = [1.0, 2.0, 3.0]
    out = downsample_sequence(seq, target_len=3, mode="mean")
    assert np.allclose(out, seq)


def test_downsample_invalid_inputs():
    with pytest.raises(ValueError, match="empty sequence"):
        downsample_sequence([], target_len=4)

    with pytest.raises(ValueError, match="target_len must be positive"):
        downsample_sequence([1, 2, 3], target_len=0)

    with pytest.raises(ValueError, match="Unknown downsample mode"):
        downsample_sequence([1, 2, 3], target_len=2, mode="median")


def test_align_features_downsampling_pooling_modes():
    # Audio of length T=100. Representations T'=4 frames at [12.5, 37.5, 62.5, 87.5].
    # Target count K=2 at [25.0, 75.0].
    # Rep frames 0 and 1 (12.5, 37.5) map to target 0 (25.0).
    # Rep frames 2 and 3 (62.5, 87.5) map to target 1 (75.0).
    from audio_benchmark.tasks.downsampling import align_features_to_targets

    reps = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
            [9.0, 50.0],
        ],
        dtype=np.float32,
    )
    target_ts = np.array([25.0, 75.0], dtype=np.float64)

    # Test "mean" pooling
    mean_aligned = align_features_to_targets(
        reps, target_ts, audio_length=100, pooling="mean"
    )
    assert mean_aligned.shape == (2, 2)
    assert np.allclose(mean_aligned[0], [2.0, 15.0])
    assert np.allclose(mean_aligned[1], [7.0, 40.0])

    # Test "max" pooling
    max_aligned = align_features_to_targets(
        reps, target_ts, audio_length=100, pooling="max"
    )
    assert np.allclose(max_aligned[0], [3.0, 20.0])
    assert np.allclose(max_aligned[1], [9.0, 50.0])

    # Test "min" pooling
    min_aligned = align_features_to_targets(
        reps, target_ts, audio_length=100, pooling="min"
    )
    assert np.allclose(min_aligned[0], [1.0, 10.0])
    assert np.allclose(min_aligned[1], [5.0, 30.0])

    # Test "median" pooling
    median_aligned = align_features_to_targets(
        reps, target_ts, audio_length=100, pooling="median"
    )
    assert np.allclose(median_aligned[0], [2.0, 15.0])
    assert np.allclose(median_aligned[1], [7.0, 40.0])


def test_align_features_upsampling_repetition():
    # Audio length T=100. Representations T'=2 at [25.0, 75.0].
    # Target count K=4 at [12.5, 37.5, 62.5, 87.5].
    # Targets 0, 1 map to rep 0; targets 2, 3 map to rep 1.
    from audio_benchmark.tasks.downsampling import align_features_to_targets

    reps = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    target_ts = np.array([12.5, 37.5, 62.5, 87.5], dtype=np.float64)

    aligned = align_features_to_targets(reps, target_ts, audio_length=100)
    assert aligned.shape == (4, 2)
    assert np.allclose(aligned[0], [10.0, 20.0])
    assert np.allclose(aligned[1], [10.0, 20.0])
    assert np.allclose(aligned[2], [30.0, 40.0])
    assert np.allclose(aligned[3], [30.0, 40.0])


def test_align_features_zero_assigned_target_fallback():
    # T=100, T'=4 at [12.5, 37.5, 62.5, 87.5].
    # Targets at [50.0, 50.1, 50.2].
    # rep 0 (12.5) -> nearest target 0 (50.0)
    # rep 1 (37.5) -> nearest target 0 (50.0)
    # rep 2 (62.5) -> nearest target 2 (50.2)
    # rep 3 (87.5) -> nearest target 2 (50.2)
    # Target 1 (50.1) receives 0 assigned representation frames.
    # Nearest rep frame in time to target 1 is rep 2 (62.5, dist 12.4 vs rep 1 dist 12.6).
    from audio_benchmark.tasks.downsampling import align_features_to_targets

    reps = np.array(
        [[1.0], [2.0], [10.0], [20.0]],
        dtype=np.float32,
    )
    target_ts = np.array([50.0, 50.1, 50.2], dtype=np.float64)

    aligned = align_features_to_targets(
        reps, target_ts, audio_length=100, pooling="mean"
    )
    assert aligned.shape == (3, 1)
    # Target 0 pools rep 0 and 1: mean(1.0, 2.0) = 1.5
    assert np.allclose(aligned[0], [1.5])
    # Target 1 falls back to single nearest rep feature: rep 2 (10.0)
    assert np.allclose(aligned[1], [10.0])
    # Target 2 pools rep 2 and 3: mean(10.0, 20.0) = 15.0
    assert np.allclose(aligned[2], [15.0])


def test_align_features_validation():
    from audio_benchmark.tasks.downsampling import align_features_to_targets

    reps = np.ones((4, 8), dtype=np.float32)
    target_ts = np.array([10.0, 20.0])

    with pytest.raises(ValueError, match="Unknown pooling"):
        align_features_to_targets(reps, target_ts, audio_length=100, pooling="invalid")

    with pytest.raises(ValueError, match="audio_length must be positive"):
        align_features_to_targets(reps, target_ts, audio_length=0)

    with pytest.raises(ValueError, match="representations must have shape"):
        align_features_to_targets(np.ones(4), target_ts, audio_length=100)


def test_parse_frame_targets_dictionary_and_sequence():
    from audio_benchmark.exceptions import InvalidDatasetError
    from audio_benchmark.tasks.downsampling import parse_frame_targets

    # Uniform sequence
    ts, vals = parse_frame_targets([10, 20, 30], audio_length=300)
    assert len(ts) == 3
    assert np.allclose(ts, [50.0, 150.0, 250.0])
    assert vals == [10, 20, 30]

    # Dictionary targets with string and int keys
    ts_dict, vals_dict = parse_frame_targets(
        {"100": 1.0, 50: 0.5}, audio_length=200, key_name="frame_targets"
    )
    # Sorted by timestamp: 50 then 100
    assert list(ts_dict) == [50.0, 100.0]
    assert vals_dict == [0.5, 1.0]

    # Error cases
    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets([], audio_length=100)

    with pytest.raises(InvalidDatasetError, match="must not be empty"):
        parse_frame_targets({}, audio_length=100)

    with pytest.raises(InvalidDatasetError, match="exceeds audio length"):
        parse_frame_targets({100: 1.0}, audio_length=100)

    with pytest.raises(InvalidDatasetError, match="must be an integer"):
        parse_frame_targets({True: 1.0}, audio_length=100)

    with pytest.raises(InvalidDatasetError, match="not a valid integer"):
        parse_frame_targets({"invalid_int": 1.0}, audio_length=100)
