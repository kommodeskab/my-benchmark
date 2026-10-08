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
  - For models operating on pooled representations (such as `LinearBenchmarkClassifier`, `KNNBenchmarkClassifier`, `MLPBenchmarkClassifier`): sequence representations `(B, T', C')` are pooled over time to `(B, C')`.
  - For models accepting sequences directly (such as `TransformerBenchmarkClassifier`): **no pooling** is applied; a non-causal Transformer encoder attends across all temporal frames directly.
- **Parameters**:
  - `classifier` (`BaseClassifier`, required): An explicit instantiated classifier model (e.g., `LinearBenchmarkClassifier()`, `KNNBenchmarkClassifier()`, `MLPBenchmarkClassifier()`, `TransformerBenchmarkClassifier()`, or `BayesOptClassifier(...)`). Passing `None` is not allowed.
  - `label_key` (`str`, default `"label"`): Key in sample metadata containing the ground truth label.
  - `pooling` (`str`, default `"mean"`): Temporal pooling method for classifiers that do not accept 3D sequence tensors (`"mean"`, `"max"`, `"first"`, `"last"`). Bypassed when using sequence models like `TransformerBenchmarkClassifier`.
  - `metric` (`str`, default `"accuracy"`): Main metric for the task (`"accuracy"`, `"macro_f1"`, `"micro_f1"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import (
    GlobalClassificationTask,
    KNNBenchmarkClassifier,
    LinearBenchmarkClassifier,
    TransformerBenchmarkClassifier,
)

# Linear classifier (Logistic Regression) with mean pooling
task_linear = GlobalClassificationTask(
    classifier=LinearBenchmarkClassifier(model_type="logistic", C=1.0),
    pooling="mean",
)

# KNN with mean pooling
task_knn = GlobalClassificationTask(
    classifier=KNNBenchmarkClassifier(n_neighbors=5),
    pooling="mean",
)

# Non-causal Transformer (no pooling, attends across all features)
task_tf = GlobalClassificationTask(classifier=TransformerBenchmarkClassifier(epochs=20))
```

---

### 2. `RegressionTask`
Predicts a continuous target value for the entire audio clip.

- **Metadata Key**: `'target'` (or custom via `target_key`).
- **How it works**:
  - For models operating on pooled representations (such as `LinearBenchmarkRegressor`, `KNNBenchmarkRegressor`, `MLPBenchmarkRegressor`): sequence representations `(B, T', C')` are pooled over time to `(B, C')`.
  - For models accepting sequences directly (such as `TransformerBenchmarkRegressor`): **no pooling** is applied; a non-causal Transformer attends across all temporal frames directly.
- **Parameters**:
  - `regressor` (`BaseRegressor`, required): An explicit instantiated regressor model (e.g., `LinearBenchmarkRegressor()`, `KNNBenchmarkRegressor()`, `MLPBenchmarkRegressor()`, `TransformerBenchmarkRegressor()`, or `BayesOptRegressor(...)`). Passing `None` is not allowed.
  - `target_key` (`str`, default `"target"`): Key in sample metadata containing the numeric target.
  - `pooling` (`str`, default `"mean"`): Temporal pooling method for linear/KNN/MLP. Bypassed for sequence models like `TransformerBenchmarkRegressor`.
  - `metric` (`str`, default `"r2"`): Main metric for the task (`"r2"`, `"mse"`, `"rmse"`, `"mae"`, `"pearson_r"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import (
    LinearBenchmarkRegressor,
    RegressionTask,
    TransformerBenchmarkRegressor,
)

task_linear = RegressionTask(regressor=LinearBenchmarkRegressor(alpha=1.0))
task_tf = RegressionTask(regressor=TransformerBenchmarkRegressor(epochs=20))
```

---

### 3. `FrameClassificationTask`
Classifies a label for each frame across time.

- **Metadata Key**: `'frame_labels'` (or custom via `label_key`). Must be a sequence of length $T$ matching the raw audio length.
- **How it works**: When the representation model downsamples audio from $T$ to $T'$, this task downsamples the ground truth labels to $T'$ by taking the most common label (mode) in each window.
- **Parameters**:
  - `classifier` (`BaseClassifier`, required): An explicit instantiated classifier model (e.g., `LinearBenchmarkClassifier()`, `KNNBenchmarkClassifier()`, or `MLPBenchmarkClassifier()`). Passing `None` is not allowed.
  - `label_key` (`str`, default `"frame_labels"`): Metadata key with the per-frame label sequence.
  - `metric` (`str`, default `"accuracy"`): Main metric for the task (`"accuracy"`, `"macro_f1"`, `"micro_f1"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import FrameClassificationTask, KNNBenchmarkClassifier

task_frame_clf = FrameClassificationTask(classifier=KNNBenchmarkClassifier())
```

---

### 4. `FrameRegressionTask`
Predicts a continuous target for each frame across time.

- **Metadata Key**: `'frame_targets'` (or custom via `target_key`). Must be a sequence of length $T$ matching raw audio length.
- **How it works**: Downsamples the continuous targets from $T$ to $T'$ by computing the mean value in each downsampled region.
- **Parameters**:
  - `regressor` (`BaseRegressor`, required): An explicit instantiated regressor model (e.g., `LinearBenchmarkRegressor()`, `KNNBenchmarkRegressor()`, or `MLPBenchmarkRegressor()`). Passing `None` is not allowed.
  - `target_key` (`str`, default `"frame_targets"`): Metadata key with the per-frame numeric sequence.
  - `metric` (`str`, default `"r2"`): Main metric for the task (`"r2"`, `"mse"`, `"rmse"`, `"mae"`, `"pearson_r"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import FrameRegressionTask, LinearBenchmarkRegressor

task_frame_reg = FrameRegressionTask(regressor=LinearBenchmarkRegressor())
```

---

### 5. `SequenceModelingTask`
Models variable-length discrete token sequences (e.g., words, phonemes, character IDs).

- **Metadata Key**: `'tokens'` (or custom via `target_key`). A list of strings (e.g. `["ph_0", "ph_1"]`) or integers. A vocabulary is automatically built if strings are provided.
- **Probe Architecture**:
  - `CTCSequenceProbe`: A linear projection probe trained with Connectionist Temporal Classification loss and greedy best-path collapse decoding.
  - `TransformerDecoderProbe`: An autoregressive Transformer decoder with cross-attention over the audio representations and causal decoding.
- **Metrics**: Token Error Rate (PER/CER via Levenshtein edit distance) and Exact Sequence Match Accuracy.
- **Parameters**:
  - `probe` (`BaseSequenceProbe`, required): An explicit probe instance (`CTCSequenceProbe(...)` or `TransformerDecoderProbe(...)`). Passing `None` or string names is not allowed.
  - `target_key` (`str`, default `"tokens"`): Metadata key containing the token sequence.
  - `metric` (`str`, default `"token_error_rate"`): Main metric for the task (`"token_error_rate"`, `"sequence_accuracy"`), also populated in `result.metrics["value"]`.

```python
from audio_benchmark import (
    CTCSequenceProbe,
    SequenceModelingTask,
    TransformerDecoderProbe,
)

# CTC linear projection probe
task_ctc = SequenceModelingTask(probe=CTCSequenceProbe(epochs=20))

# Autoregressive Transformer decoder probe
task_tf = SequenceModelingTask(
    probe=TransformerDecoderProbe(epochs=20, d_model=64, nhead=4)
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
    KNNBenchmarkClassifier,
    LinearBenchmarkRegressor,
    RegressionTask,
    SequenceModelingTask,
    TransformerDecoderProbe,
)

# 1. Provide your model and datasets
model = DummyAudioModel(representation_dim=64)
train_dataset = DummyAudioDataset(
    num_samples=40, audio_length=16000, num_classes=4, name="speech_train"
)
test_dataset = DummyAudioDataset(
    num_samples=16, audio_length=16000, num_classes=4, name="speech_test"
)

# 2. Initialize benchmarks with datasets, task, and cache settings
bench_clf = AudioBenchmark(
    dataset=train_dataset,
    testset=test_dataset,
    task=GlobalClassificationTask(classifier=KNNBenchmarkClassifier()),
    cache_dir=".benchmark_cache",
    enable_cache=True,
)
res_clf = bench_clf.run(model=model, model_name="my_model_v1")

# Reuses cached representations automatically across tasks!
bench_reg = AudioBenchmark(
    dataset=train_dataset,
    testset=test_dataset,
    task=RegressionTask(regressor=LinearBenchmarkRegressor()),
    cache_dir=".benchmark_cache",
    enable_cache=True,
)
res_reg = bench_reg.run(model=model, model_name="my_model_v1")

bench_seq = AudioBenchmark(
    dataset=train_dataset,
    testset=test_dataset,
    task=SequenceModelingTask(probe=TransformerDecoderProbe(epochs=10)),
    cache_dir=".benchmark_cache",
    enable_cache=True,
)
res_seq = bench_seq.run(model=model, model_name="my_model_v1")

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

## Bayesian Optimization for Downstream Models

You can optimize downstream classifier and regressor hyperparameters via Bayesian optimization using [`BayesOptClassifier`](file:///home/andbagge/my-benchmark/audio_benchmark/bayes_opt.py) and [`BayesOptRegressor`](file:///home/andbagge/my-benchmark/audio_benchmark/bayes_opt.py).

The wrappers use [Optuna](https://optuna.org/) to run K-fold cross-validation on the training set, select the optimal hyperparameter combination, and evaluate only the best-performing model on the downstream test set. Progress is displayed in real time using `tqdm`.

### Quick Example

```python
from audio_benchmark import (
    AudioBenchmark,
    BayesOptClassifier,
    BayesOptRegressor,
    GlobalClassificationTask,
    KNNBenchmarkClassifier,
    MLPBenchmarkClassifier,
    RegressionTask,
)

# Optimize KNN classifier hyperparameters (e.g. n_neighbors, distance metric)
knn_space = {
    "n_neighbors": (1, 15),
    "metric": ["minkowski", "cosine", "euclidean"],
}
opt_clf = BayesOptClassifier(
    base_model=KNNBenchmarkClassifier(), param_space=knn_space, n_trials=15, cv=3
)
task_clf = GlobalClassificationTask(classifier=opt_clf)

benchmark = AudioBenchmark(
    dataset=train_dataset,
    testset=test_dataset,
    task=task_clf,
)
result = benchmark.run(model=my_model, model_name="encoder")

# Benchmark reports standard metrics for the best model plus best hyperparameters:
result.print_summary()
print("Best hyperparameters found:", result.metrics["best_params"])
print("Best CV score:", result.metrics["best_cv_score"])

# Custom parameter spaces (handling discrete choices, integer ranges, and continuous values):
custom_space = {
    "hidden_dim": [64, 128, 256],  # Discrete categorical choices
    "lr": (1e-4, 1e-2, "log"),  # Continuous log-uniform float
    "dropout": (0.0, 0.4),  # Continuous uniform float
}
opt_mlp = BayesOptClassifier(
    base_model=MLPBenchmarkClassifier(), param_space=custom_space, n_trials=10
)
```

---

## Running Tests

To verify the test suite:

```bash
uv run pytest -v
```
