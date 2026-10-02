"""Quickstart example demonstrating how to run the audio benchmark suite in Python.

This script demonstrates:
1. Running multiple benchmark tasks silently (no internal printing from benchmark.run).
2. Attaining all result objects.
3. Printing the results explicitly by the user (using res.print_summary() and custom formatting).
"""

from audio_benchmark import (
    AudioBenchmark,
    DummyAudioDataset,
    DummyAudioModel,
    FrameClassificationTask,
    FrameRegressionTask,
    GlobalClassificationTask,
    KNNBenchmarkClassifier,
    LinearBenchmarkRegressor,
    RegressionTask,
    SequenceModelingTask,
    TransformerBenchmarkClassifier,
    TransformerBenchmarkRegressor,
)


def main():
    print("=== Initializing Model, Dataset, and Benchmark ===")
    model = DummyAudioModel(representation_dim=64)
    model_name = "conv_encoder_v1"

    dataset = DummyAudioDataset(
        num_samples=40,
        audio_length=16000,  # 1 second of audio at 16kHz
        num_classes=4,
        name="synthetic_speech_v1",
    )

    benchmark = AudioBenchmark(cache_dir=".benchmark_cache", enable_cache=True)

    print("\nExecuting benchmark tasks...")
    # Step 1: Attain all results silently from benchmark.run()
    result_global_clf = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=GlobalClassificationTask(classifier=KNNBenchmarkClassifier()),
        batch_size=16,
    )

    result_global_reg = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=RegressionTask(regressor=LinearBenchmarkRegressor()),
    )

    result_global_clf_tf = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=GlobalClassificationTask(
            classifier=TransformerBenchmarkClassifier(epochs=15)
        ),
    )

    result_global_reg_tf = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=RegressionTask(regressor=TransformerBenchmarkRegressor(epochs=15)),
    )

    result_frame_clf = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=FrameClassificationTask(classifier=KNNBenchmarkClassifier()),
    )

    result_frame_reg = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=FrameRegressionTask(regressor=LinearBenchmarkRegressor()),
    )

    result_seq_ctc = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=SequenceModelingTask(method="ctc", epochs=15),
    )

    result_seq_tf = benchmark.run(
        model=model,
        model_name=model_name,
        dataset=dataset,
        task=SequenceModelingTask(method="transformer", epochs=15),
    )
    print("All tasks finished successfully.\n")

    # Step 2: The user decides how and when to print the results
    print("=== Detailed Results (via result.print_summary()) ===")
    all_results = [
        result_global_clf,
        result_global_clf_tf,
        result_global_reg,
        result_global_reg_tf,
        result_frame_clf,
        result_frame_reg,
        result_seq_ctc,
        result_seq_tf,
    ]
    for result in all_results:
        result.print_summary()

    # Step 3: Custom user-defined summary
    print("\n=== User-Defined Summary Table ===")
    print(
        f"1. Global Clf (KNN):           Acc: {result_global_clf.metrics['accuracy'] * 100:.1f}% | Train Time: {result_global_clf.metrics['training_time']:.4f}s (Cached: {result_global_clf.cached})"
    )
    print(
        f"2. Global Clf (Transformer):   Acc: {result_global_clf_tf.metrics['accuracy'] * 100:.1f}% | Train Time: {result_global_clf_tf.metrics['training_time']:.4f}s (Cached: {result_global_clf_tf.cached})"
    )
    print(
        f"3. Global Reg (Linear):        R²:  {result_global_reg.metrics['r2']:.4f} | Train Time: {result_global_reg.metrics['training_time']:.4f}s (Cached: {result_global_reg.cached})"
    )
    print(
        f"4. Global Reg (Transformer):   R²:  {result_global_reg_tf.metrics['r2']:.4f} | Train Time: {result_global_reg_tf.metrics['training_time']:.4f}s (Cached: {result_global_reg_tf.cached})"
    )
    print(
        f"5. Frame Clf (KNN):            Acc: {result_frame_clf.metrics['accuracy'] * 100:.1f}% | Train Time: {result_frame_clf.metrics['training_time']:.4f}s (Cached: {result_frame_clf.cached})"
    )
    print(
        f"6. Frame Reg (Linear):         R²:  {result_frame_reg.metrics['r2']:.4f} | Train Time: {result_frame_reg.metrics['training_time']:.4f}s (Cached: {result_frame_reg.cached})"
    )
    print(
        f"7. Sequence (CTC):             TER: {result_seq_ctc.metrics['token_error_rate'] * 100:.1f}% | Train Time: {result_seq_ctc.metrics['training_time']:.4f}s (Cached: {result_seq_ctc.cached})"
    )
    print(
        f"8. Sequence (Transformer):     TER: {result_seq_tf.metrics['token_error_rate'] * 100:.1f}% | Train Time: {result_seq_tf.metrics['training_time']:.4f}s (Cached: {result_seq_tf.cached})"
    )

    # Step 4: Instant key metric extraction across all tasks via metrics["value"]
    print("\n=== Instant Key Metric Extraction via result.metrics['value'] ===")
    for res in all_results:
        print(f"  {res.task_name:22s} -> value: {res.metrics['value']:.4f}")


if __name__ == "__main__":
    main()
