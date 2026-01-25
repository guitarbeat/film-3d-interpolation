## 2026-01-13 - [CPU-bound Model Inference Optimization]
**Learning:** Wrapping TensorFlow Hub model calls in `@tf.function(jit_compile=True)` on a CPU-only environment provided only a marginal speedup (~2%) for the FILM model.
**Action:** While XLA compilation (jit_compile=True) is generally powerful for fusing ops, its impact on heavy pre-trained models on CPU can be limited if the bottleneck is pure compute intensity rather than graph overhead. However, it is still a valid best practice for deployment. Be cautious of compilation overhead during the first call (warmup).

## 2026-01-19 - [XLA Compilation Overhead on CPU]
**Learning:** Enabling XLA (`jit_compile=True`) for the FILM model on CPU introduced a significant startup latency (~3s overhead) due to compilation, while only offering negligible improvements for subsequent steady-state inference.
**Action:** For CLI tools or applications with short lifecycles running on CPU, disable JIT compilation (`jit_compile=False`) to prioritize faster startup and immediate responsiveness.

## 2026-01-25 - [Graph Execution Compatibility]
**Learning:** Using `np.repeat` on a Tensor forces an implicit conversion to NumPy, which raises `NotImplementedError` when attempting to trace the function with `@tf.function` (graph execution).
**Action:** Use `tf.repeat` instead of `np.repeat` when working with potential Tensors. `tf.repeat` is compatible with both Tensors and NumPy arrays (converting them to Tensors), ensuring the operation remains graph-compatible.
