# Audio Representation Benchmark Suite

A lightweight, modular Python toolkit for benchmarking audio representation learning models on downstream audio tasks.

---

## Installation (Editable Mode)

To use this benchmark suite in your own project or environment, install it in editable mode (`-e`):

```bash
# Using pip:
pip install -e /path/to/my-benchmark

# Or using uv:
uv add --editable /path/to/my-benchmark
# or
uv pip install -e /path/to/my-benchmark
```

Once installed, you can import `audio_benchmark` directly in any Python script:

```python
import audio_benchmark
```

---

## What is Audio Benchmark?

- **Multiple Downstream Tasks**: Evaluate representations on global classification/regression, frame-level classification/regression, and sequence modeling.
- **Fast Downstream Probes**: Choose between Linear/Ridge, KNN, MLP, and non-causal Transformer models.
- **Automatic Memory-Mapped Caching**: Audio representations are streamed directly to disk into memory-mapped NumPy/PyTorch arrays, scaling to massive datasets with constant, low RAM usage.
- **Training Time Tracking**: Measures the exact training duration (`training_time`) for each downstream task.
- **100% Python API**: Call and configure tasks directly in Python code.

---

## Dataset & Model Requirements

### 1. Representation Model
Any `torch.nn.Module` that accepts raw audio of shape `(B, 1, T)` and returns representations of shape `(B, T', C')`:
- `B`: Batch size
- `T'`: Downsampled representation sequence length
- `C'`: Representation feature dimension

### 2. Audio Dataset
Your PyTorch `Dataset.__getitem__` must return a dictionary with three keys:
```python
{
    "waveform": audio_tensor,  # torch.Tensor of shape (1, T) or (T,)
    "sample_rate": sample_rate,  # int or float (e.g. 16000)
    "metadata": metadata_dict,  # standard Python dict with ground truth targets
}
```
- `"waveform"`: A 1D `(T,)` or 2D mono `(1, T)` `torch.Tensor` containing raw audio samples.
- `"sample_rate"`: Sampling rate as a positive integer or float (e.g. `16000`).
- `"metadata"`: A standard Python `dict` containing the ground truth targets for the task.

If your raw dataset has a different format, wrap it:
```python
import torch
from torch.utils.data import Dataset


class MyAudioDatasetWrapper(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        return {
            "waveform": torch.as_tensor(item["audio"], dtype=torch.float32),
            "sample_rate": item.get("sample_rate", 16000),
            "metadata": {
                "label": item["class_id"],  # For global classification
                "target": item["pitch_val"],  # For global regression
                "frame_labels": item[
                    "frame_ids"
                ],  # For frame classification (length T)
                "frame_targets": item["frame_vals"],  # For frame regression (length T)
                "tokens": item["phonemes"],  # For sequence modeling (list of tokens)
            },
        }
```

---

## Available Benchmark Tasks

### 1. `GlobalClassificationTask`
Classifies a single categorical label for the entire audio clip.

- **Metadata Key**: `'label'` (or custom via `label_key`).
- **How it works**:
  - For `'linear'`, `'knn'`, and `'mlp'`: sequence representations `(B, T', C')` are pooled over time to `(B, C')`.
  - For `'transformer'`: **no pooling** is applied; a non-causal Transformer encoder attends across all temporal frames directly.
- **Parameters**:
  - `classifier` (`str | BaseClassifier`, default `"knn"`): `'linear'`, `'knn'`, `'mlp'`, or `'transformer'`.
  - `label_key` (`str`, default `"label"`): Key in sample metadata containing the ground truth label.
  - `pooling` (`str`, default `"mean"`): Temporal pooling method for `'linear'`, `'knn'`, and `'mlp'` (`"mean"`, `"max"`, `"first"`, `"last"`). Bypassed when using `'transformer'`.
  - `test_size` (`float`, default `0.2`): Fraction of dataset reserved for testing.
  - `split_key` (`str | None`, default `None`): Optional metadata key specifying pre-defined splits (`"train"` / `"test"`).
  - `classifier_kwargs` (`dict | None`): Hyperparameters passed to the classifier (e.g., `C` for linear logistic; `n_neighbors` for KNN; `epochs`, `lr`, `d_model`, `nhead` for MLP/Transformer).
  - `metric` (`str`, default `"accuracy"`): Main metric for the task (`"accuracy"`, `"macro_f1"`, `"micro_f1"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import GlobalClassificationTask

# Linear classifier (Logistic Regression) with mean pooling
task_linear = GlobalClassificationTask(classifier="linear", pooling="mean")

# KNN with mean pooling
task_knn = GlobalClassificationTask(classifier="knn", pooling="mean")

# Non-causal Transformer (no pooling, attends across all features)
task_tf = GlobalClassificationTask(
    classifier="transformer", classifier_kwargs={"epochs": 20}
)
```

---

### 2. `RegressionTask`
Predicts a continuous target value for the entire audio clip.

- **Metadata Key**: `'target'` (or custom via `target_key`).
- **How it works**:
  - For `'linear'`, `'knn'`, and `'mlp'`: sequence representations `(B, T', C')` are pooled over time to `(B, C')`.
  - For `'transformer'`: **no pooling** is applied; a non-causal Transformer attends across all temporal frames directly.
- **Parameters**:
  - `regressor` (`str | BaseRegressor`, default `"linear"`): `'linear'`, `'knn'`, `'mlp'`, or `'transformer'`.
  - `target_key` (`str`, default `"target"`): Key in sample metadata containing the numeric target.
  - `pooling` (`str`, default `"mean"`): Temporal pooling method for linear/KNN/MLP. Bypassed for `'transformer'`.
  - `test_size` (`float`, default `0.2`): Fraction of dataset reserved for testing.
  - `split_key` (`str | None`, default `None`): Optional metadata key for pre-defined train/test splits.
  - `regressor_kwargs` (`dict | None`): Hyperparameters for the regressor (e.g., `alpha` for linear ridge; `epochs`, `lr` for MLP/Transformer).
  - `metric` (`str`, default `"r2"`): Main metric for the task (`"r2"`, `"mse"`, `"rmse"`, `"mae"`, `"pearson_r"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import RegressionTask

task_linear = RegressionTask(regressor="linear")
task_tf = RegressionTask(regressor="transformer", regressor_kwargs={"epochs": 20})
```

---

### 3. `FrameClassificationTask`
Classifies a label for each frame across time.

- **Metadata Key**: `'frame_labels'` (or custom via `label_key`). Must be a sequence of length $T$ matching the raw audio length.
- **How it works**: When the representation model downsamples audio from $T$ to $T'$, this task downsamples the ground truth labels to $T'$ by taking the most common label (mode) in each window.
- **Parameters**:
  - `classifier` (`str | BaseClassifier`, default `"knn"`): `'linear'`, `'knn'`, or `'mlp'`.
  - `label_key` (`str`, default `"frame_labels"`): Metadata key with the per-frame label sequence.
  - `test_size` (`float`, default `0.2`): Train/test split at the audio clip level (avoids temporal leakage).
  - `split_key` (`str | None`, default `None`): Optional metadata key for pre-defined splits.
  - `classifier_kwargs` (`dict | None`): Hyperparameters passed to the classifier.
  - `metric` (`str`, default `"accuracy"`): Main metric for the task (`"accuracy"`, `"macro_f1"`, `"micro_f1"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import FrameClassificationTask

task_frame_clf = FrameClassificationTask(classifier="knn")
```

---

### 4. `FrameRegressionTask`
Predicts a continuous target for each frame across time.

- **Metadata Key**: `'frame_targets'` (or custom via `target_key`). Must be a sequence of length $T$ matching raw audio length.
- **How it works**: Downsamples the continuous targets from $T$ to $T'$ by computing the mean value in each downsampled region.
- **Parameters**:
  - `regressor` (`str | BaseRegressor`, default `"linear"`): `'linear'`, `'knn'`, or `'mlp'`.
  - `target_key` (`str`, default `"frame_targets"`): Metadata key with the per-frame numeric sequence.
  - `test_size` (`float`, default `0.2`): Train/test split at the clip level.
  - `split_key` (`str | None`, default `None`): Optional metadata key for pre-defined splits.
  - `regressor_kwargs` (`dict | None`): Hyperparameters passed to the regressor.
  - `metric` (`str`, default `"r2"`): Main metric for the task (`"r2"`, `"mse"`, `"rmse"`, `"mae"`, `"pearson_r"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import FrameRegressionTask

task_frame_reg = FrameRegressionTask(regressor="linear")
```

---

### 5. `SequenceModelingTask`
Models variable-length discrete token sequences (e.g., words, phonemes, character IDs).

- **Metadata Key**: `'tokens'` (or custom via `target_key`). A list of strings (e.g. `["ph_0", "ph_1"]`) or integers. A vocabulary is automatically built if strings are provided.
- **Methods**:
  - `'ctc'`: A linear projection probe trained with Connectionist Temporal Classification loss and greedy best-path collapse decoding.
  - `'transformer'`: An autoregressive Transformer decoder with cross-attention over the audio representations and causal decoding.
- **Metrics**: Token Error Rate (PER/CER via Levenshtein edit distance) and Exact Sequence Match Accuracy.
- **Parameters**:
  - `method` (`str`, default `"ctc"`): Modeling architecture (`"ctc"` or `"transformer"`).
  - `target_key` (`str`, default `"tokens"`): Metadata key containing the token sequence.
  - `epochs` (`int`, default `25`): Number of training epochs for the downstream probe.
  - `batch_size` (`int`, default `16`): Batch size for probe training.
  - `lr` (`float`, default `1e-3`): Learning rate for probe optimizer.
  - `test_size` (`float`, default `0.2`): Clip-level train/test split.
  - `split_key` (`str | None`, default `None`): Optional metadata key for pre-defined splits.
  - `model_kwargs` (`dict | None`): Extra kwargs for the probe (e.g. `d_model`, `nhead`, `num_layers` for Transformer decoder).
  - `metric` (`str`, default `"token_error_rate"`): Main metric for the task (`"token_error_rate"`, `"sequence_accuracy"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import SequenceModelingTask

# CTC linear projection probe
task_ctc = SequenceModelingTask(method="ctc", epochs=20)

# Autoregressive Transformer decoder probe
task_tf = SequenceModelingTask(method="transformer", epochs=20)
```

---

## Dynamic Metadata Wrappers

You can dynamically annotate existing audio datasets with metadata labels required by benchmark tasks. Wrappers can be used independently or chained together, and include built-in in-memory caching (`cache=True`) so annotations are computed only once per sample.

### 1. `WhisperTranscriptionWrapper`
Adds text transcriptions using the OpenAI Whisper model (defaults to `"small"`). Audio of any sample rate is automatically resampled to 16 kHz.

- **Added Key**: `'transcription'` (stored in `item["metadata"]`).
- **Parameters**:
  - `model_name` (`str`, default `"small"`): Whisper model variant (`"small"`, `"base"`, `"tiny"`).
  - `device` (`str | None`, default `None`): Device to run Whisper on (`"cuda"` if available, else `"cpu"`).
  - `language` (`str | None`, default `None`): Optional language code (e.g. `"en"` or auto-detect).
  - `key` (`str`, default `"transcription"`): Target metadata key.
  - `cache` (`bool`, default `True`): In-memory caching for generated transcriptions.

```python
from audio_benchmark import WhisperTranscriptionWrapper

dataset_with_text = WhisperTranscriptionWrapper(dataset, model_name="small")
```

### 2. `VoiceActivityWrapper`
Generates sample-by-sample voice activity detection (VAD) labels for each input in the raw waveform. For a waveform of length $T$, it produces a binary sequence of length $T$ (1 if speaking, 0 if silent), directly compatible with `FrameClassificationTask`.

- **Added Key**: `'frame_labels'` (or custom via `key`).
- **Parameters**:
  - `method` (`str`, default `"silero"`): `"silero"` (deep learning Silero VAD) or `"energy"` (short-time energy detector).
  - `threshold` (`float`, default `0.5`): Speech detection sensitivity threshold.
  - `key` (`str`, default `"frame_labels"`): Target metadata key.
  - `cache` (`bool`, default `True`): In-memory caching.

```python
from audio_benchmark import VoiceActivityWrapper

# Adds binary frame labels (length T) matching raw audio waveform length
dataset_with_vad = VoiceActivityWrapper(dataset, method="silero")
```

### 3. `PhonemeWrapper`
Generates discrete phoneme token sequences (e.g. `['HH', 'AH0', 'L', 'OW1']`) compatible with `SequenceModelingTask`. Converts text transcriptions via Grapheme-to-Phoneme (`g2p_en`). If text is not already present in the dataset, it automatically transcribes the audio waveform using Whisper first.

- **Added Key**: `'tokens'` (or custom via `key`).
- **Parameters**:
  - `key` (`str`, default `"tokens"`): Target metadata key for the phoneme sequence.
  - `text_key` (`str`, default `"transcription"`): Metadata key to check for existing text.
  - `drop_punctuation` (`bool`, default `True`): Filter out non-phonemic punctuation marks.
  - `strip_stress` (`bool`, default `False`): Strip ARPAbet stress numbers (e.g. `'AH0'` -> `'AH'`).
  - `cache` (`bool`, default `True`): In-memory caching.

```python
from audio_benchmark import PhonemeWrapper

# Transcribes with Whisper (if text absent) and converts to phonemes
dataset_with_phonemes = PhonemeWrapper(dataset, drop_punctuation=True)
```

### Chaining Wrappers
Wrappers can be chained together:

```python
dataset_annotated = VoiceActivityWrapper(
    PhonemeWrapper(WhisperTranscriptionWrapper(dataset, model_name="small"))
)
```

---

## Quickstart Example

Here is a complete, minimal example using dummy models and datasets included with the package:

```python
from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    GlobalClassificationTask,
    RegressionTask,
    SequenceModelingTask,
)

# 1. Provide your model and dataset
model = DummyAudioModel(representation_dim=64)
dataset = DummyAudioDataset(num_samples=40, audio_length=16000, num_classes=4)

# 2. Initialize benchmark runner with representation caching enabled
benchmark = AudioBenchmark(cache_dir=".benchmark_cache", enable_cache=True)

# 3. Run tasks (runs silently and returns result objects)
res_clf = benchmark.run(
    model=model,
    model_name="my_model_v1",
    dataset=dataset,
    task=GlobalClassificationTask(classifier="knn"),
)

# Reuses cached representations automatically!
res_reg = benchmark.run(
    model=model,
    model_name="my_model_v1",
    dataset=dataset,
    task=RegressionTask(regressor="linear"),
)

res_seq = benchmark.run(
    model=model,
    model_name="my_model_v1",
    dataset=dataset,
    task=SequenceModelingTask(method="transformer", epochs=10),
)

# 4. Print summaries or inspect metrics
for res in [res_clf, res_reg, res_seq]:
    res.print_summary()

# Every BenchmarkResult guarantees a "value" key containing the main task metric:
for res in [res_clf, res_reg, res_seq]:
    print(f"{res.task_name:22s} -> Main Result (value): {res.metrics['value']}")

# The metric is also accessible under its specific name:
print(f"Classification accuracy: {res_clf.metrics['accuracy'] * 100:.1f}%")
print(f"Downstream probe training time: {res_clf.metrics['training_time']:.4f}s")
```

---

## Running Tests

To verify the test suite:

```bash
uv run pytest -v
```
