"""Tests for representation caching mechanism."""

import pytest
import torch

from audio_benchmark.cache import RepresentationCache


def test_cache_save_and_load(tmp_path):
    cache = RepresentationCache(cache_dir=tmp_path)
    model_name = "test_model_v1"
    dataset_name = "test_dataset"

    assert not cache.has(model_name, dataset_name)

    reps = [torch.randn(20, 15, 64), torch.randn(20, 10, 32)]
    metadata = [{"label": i % 3, "sample_id": i} for i in range(20)]

    saved_path = cache.save(model_name, dataset_name, reps, metadata)
    assert saved_path.exists()
    assert cache.has(model_name, dataset_name)

    loaded_reps, loaded_meta = cache.load(model_name, dataset_name)
    assert isinstance(loaded_reps, list)
    assert len(loaded_reps) == 2
    assert torch.allclose(reps[0], loaded_reps[0])
    assert torch.allclose(reps[1], loaded_reps[1])
    assert len(loaded_meta) == 20
    assert loaded_meta[5]["label"] == metadata[5]["label"]


def test_cache_load_nonexistent_raises(tmp_path):
    cache = RepresentationCache(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.load("non_existent_model", "non_existent_dataset")


def test_cache_clear(tmp_path):
    cache = RepresentationCache(cache_dir=tmp_path)
    reps = [torch.randn(5, 10, 32)]
    meta = [{"label": 0} for _ in range(5)]

    cache.save("model1", "dataA", reps, meta)
    cache.save("model1", "dataB", reps, meta)
    cache.save("model2", "dataA", reps, meta)

    assert cache.has("model1", "dataA")
    assert cache.has("model1", "dataB")
    assert cache.has("model2", "dataA")

    # Clear specific dataset
    cache.clear("model1", "dataA")
    assert not cache.has("model1", "dataA")
    assert cache.has("model1", "dataB")

    # Clear specific model
    cache.clear("model1")
    assert not cache.has("model1", "dataB")
    assert cache.has("model2", "dataA")

    # Clear all
    cache.clear()
    assert not cache.has("model2", "dataA")


def test_cache_memmap_files_and_manifest(tmp_path):
    cache = RepresentationCache(cache_dir=tmp_path)
    model_name = "encoder_large"
    dataset_name = "audio_set_10k"

    reps = [torch.randn(15, 8, 32), torch.randn(15, 4, 16)]
    meta = [{"idx": i} for i in range(15)]

    entry_dir = cache.save(model_name, dataset_name, reps, meta)
    assert entry_dir.is_dir()
    assert (entry_dir / "representations_0.npy").is_file()
    assert (entry_dir / "representations_1.npy").is_file()
    assert (entry_dir / "metadata.pt").is_file()
    assert (entry_dir / "manifest.json").is_file()

    # Load and verify it is backed by memory-mapping
    loaded_reps, loaded_meta = cache.load(model_name, dataset_name)
    assert isinstance(loaded_reps, list)
    assert len(loaded_reps) == 2
    assert loaded_reps[0].shape == (15, 8, 32)
    assert loaded_reps[1].shape == (15, 4, 16)
    assert len(loaded_meta) == 15
    assert torch.allclose(reps[0], loaded_reps[0])
    assert torch.allclose(reps[1], loaded_reps[1])

    # Test copy-on-write behavior (modifying loaded tensor does not corrupt disk)
    loaded_reps[0][0, 0, 0] = 999.0
    fresh_reps, _ = cache.load(model_name, dataset_name)
    assert fresh_reps[0][0, 0, 0] != 999.0
