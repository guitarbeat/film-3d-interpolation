## 2026-01-13 - [CPU-bound Model Inference Optimization]
**Learning:** Wrapping TensorFlow Hub model calls in `@tf.function(jit_compile=True)` on a CPU-only environment provided only a marginal speedup (~2%) for the FILM model.
**Action:** While XLA compilation (jit_compile=True) is generally powerful for fusing ops, its impact on heavy pre-trained models on CPU can be limited if the bottleneck is pure compute intensity rather than graph overhead. However, it is still a valid best practice for deployment. Be cautious of compilation overhead during the first call (warmup).

## 2026-01-18 - [XLA Compilation Overhead on CPU]
**Learning:** Enabling `jit_compile=True` for the FILM model on CPU introduced significant compilation overhead (~3.5s) on the first call without improving subsequent inference speeds (both ~3.1s).
**Action:** Disable `jit_compile` for this specific model/environment combination to improve startup time. Always benchmark "first time to inference" in addition to steady-state throughput, especially for CLI tools where startup latency matters.
