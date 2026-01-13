## 2026-01-13 - [CPU-bound Model Inference Optimization]
**Learning:** Wrapping TensorFlow Hub model calls in `@tf.function(jit_compile=True)` on a CPU-only environment provided only a marginal speedup (~2%) for the FILM model.
**Action:** While XLA compilation (jit_compile=True) is generally powerful for fusing ops, its impact on heavy pre-trained models on CPU can be limited if the bottleneck is pure compute intensity rather than graph overhead. However, it is still a valid best practice for deployment. Be cautious of compilation overhead during the first call (warmup).
