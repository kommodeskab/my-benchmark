"""Quickstart example demonstrating how to run the audio benchmark suite in Python.

This script demonstrates:
1. Running multiple benchmark tasks silently (no internal printing from benchmark.run).
2. Attaining all result objects.
3. Printing the results explicitly by the user (using res.print_summary() and custom formatting).
"""

from audio_benchmark import (
    AudioBenchmark,
    CTCSequenceProbe,
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
    TransformerDecoderProbe,
)


def main():
    print("=== Initializing Model, Dataset, and Benchmark ===")
    model = DummyAudioModel(representation_dim=64)
    model_name = "conv_encoder_v1"

    dataset = DummyAudioDataset(
        num_samples=40,
        audio_length=16000,  # 1 second of audio at 16kHz
        num_classes=4,
        name="synthetic_speech_train",
    )
    testset = DummyAudioDataset(
        num_samples=16,
        audio_length=16000,
        num_classes=4,
        name="synthetic_speech_test",
    )

    print("\nExecuting benchmark tasks...")
    # Step 1: Attain all results silently from benchmark.run()
    results_global_clf = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=GlobalClassificationTask(classifier=KNNBenchmarkClassifier()),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name, batch_size=16)

    results_global_reg = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=RegressionTask(regressor=LinearBenchmarkRegressor()),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)

    results_global_clf_tf = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=GlobalClassificationTask(
            classifier=TransformerBenchmarkClassifier(epochs=15)
        ),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)

    results_global_reg_tf = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=RegressionTask(regressor=TransformerBenchmarkRegressor(epochs=15)),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)

    results_frame_clf = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=FrameClassificationTask(classifier=KNNBenchmarkClassifier()),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)

    results_frame_reg = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=FrameRegressionTask(regressor=LinearBenchmarkRegressor()),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)

    results_seq_ctc = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=SequenceModelingTask(probe=CTCSequenceProbe(epochs=15)),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)

    results_seq_tf = AudioBenchmark(
        dataset=dataset,
        testset=testset,
        task=SequenceModelingTask(probe=TransformerDecoderProbe(epochs=15)),
        dataset_name="synthetic_speech",
        testset_name="synthetic_speech_test",
        cache_dir=".benchmark_cache",
        enable_cache=True,
    ).run(model=model, model_name=model_name)
    print("All tasks finished successfully.\n")

    # Step 2: The user decides how and when to print the results
    print("=== Detailed Results (via result.print_summary()) ===")
    all_results = (
        results_global_clf
        + results_global_clf_tf
        + results_global_reg
        + results_global_reg_tf
        + results_frame_clf
        + results_frame_reg
        + results_seq_ctc
        + results_seq_tf
    )
    for result in all_results:
        result.print_summary()

    # Step 3: Custom user-defined summary for primary representation (index 0)
    print("\n=== User-Defined Summary Table (Primary Rep 0) ===")
    r_g_clf = results_global_clf[0]
    r_g_clf_tf = results_global_clf_tf[0]
    r_g_reg = results_global_reg[0]
    r_g_reg_tf = results_global_reg_tf[0]
    r_f_clf = results_frame_clf[0]
    r_f_reg = results_frame_reg[0]
    r_s_ctc = results_seq_ctc[0]
    r_s_tf = results_seq_tf[0]

    print(
        f"1. Global Clf (KNN):           Acc: {r_g_clf.metrics['accuracy'] * 100:.1f}% | Train Time: {r_g_clf.metrics['training_time']:.4f}s (Cached: {r_g_clf.cached})"
    )
    print(
        f"2. Global Clf (Transformer):   Acc: {r_g_clf_tf.metrics['accuracy'] * 100:.1f}% | Train Time: {r_g_clf_tf.metrics['training_time']:.4f}s (Cached: {r_g_clf_tf.cached})"
    )
    print(
        f"3. Global Reg (Linear):        R²:  {r_g_reg.metrics['r2']:.4f} | Train Time: {r_g_reg.metrics['training_time']:.4f}s (Cached: {r_g_reg.cached})"
    )
    print(
        f"4. Global Reg (Transformer):   R²:  {r_g_reg_tf.metrics['r2']:.4f} | Train Time: {r_g_reg_tf.metrics['training_time']:.4f}s (Cached: {r_g_reg_tf.cached})"
    )
    print(
        f"5. Frame Clf (KNN):            Acc: {r_f_clf.metrics['accuracy'] * 100:.1f}% | Train Time: {r_f_clf.metrics['training_time']:.4f}s (Cached: {r_f_clf.cached})"
    )
    print(
        f"6. Frame Reg (Linear):         R²:  {r_f_reg.metrics['r2']:.4f} | Train Time: {r_f_reg.metrics['training_time']:.4f}s (Cached: {r_f_reg.cached})"
    )
    print(
        f"7. Sequence (CTC):             TER: {r_s_ctc.metrics['token_error_rate'] * 100:.1f}% | Train Time: {r_s_ctc.metrics['training_time']:.4f}s (Cached: {r_s_ctc.cached})"
    )
    print(
        f"8. Sequence (Transformer):     TER: {r_s_tf.metrics['token_error_rate'] * 100:.1f}% | Train Time: {r_s_tf.metrics['training_time']:.4f}s (Cached: {r_s_tf.cached})"
    )

    # Step 4: Instant key metric extraction across all tasks via metrics["value"]
    print("\n=== Instant Key Metric Extraction via result.metrics['value'] ===")
    for res in all_results:
        print(f"  {res.task_name:22s} -> value: {res.metrics['value']:.4f}")


if __name__ == "__main__":
    main()
