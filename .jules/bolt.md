## 2026-01-13 - [CPU-bound Model Inference Optimization]
**Learning:** Wrapping TensorFlow Hub model calls in `@tf.function(jit_compile=True)` on a CPU-only environment provided only a marginal speedup (~2%) for the FILM model.
**Action:** While XLA compilation (jit_compile=True) is generally powerful for fusing ops, its impact on heavy pre-trained models on CPU can be limited if the bottleneck is pure compute intensity rather than graph overhead. However, it is still a valid best practice for deployment. Be cautious of compilation overhead during the first call (warmup).

## 2026-01-22 - [Tensor Pipeline Optimization]
**Learning:** Avoiding implicit NumPy conversions in TensorFlow pipelines yields massive speedups (19.5x reduction in return overhead). Additionally, `tf.reduce_max` is ~2.87x faster than `np.max` on large 5D volumes even on CPU.
**Action:** Always provide an option to return `tf.Tensor` from library functions (e.g., `return_tensor=True`) to allow users to chain TensorFlow operations without incurring expensive CPU-GPU/Python-C++ data copy overhead.
