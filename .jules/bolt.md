## 2026-01-13 - [CPU-bound Model Inference Optimization]
**Learning:** Wrapping TensorFlow Hub model calls in `@tf.function(jit_compile=True)` on a CPU-only environment provided only a marginal speedup (~2%) for the FILM model.
**Action:** While XLA compilation (jit_compile=True) is generally powerful for fusing ops, its impact on heavy pre-trained models on CPU can be limited if the bottleneck is pure compute intensity rather than graph overhead. However, it is still a valid best practice for deployment. Be cautious of compilation overhead during the first call (warmup).

## 2026-01-20 - [JIT Compilation on CPU]
**Learning:** Contrary to previous belief, `jit_compile=True` provides a ~10% speedup (5.8s vs 6.5s) for steady-state FILM inference on CPU, despite the ~3s additional warmup cost. It should be kept enabled for production workloads where throughput matters more than first-inference latency.
**Action:** Keep `jit_compile=True` enabled.

## 2026-01-20 - [Tensor vs Numpy Reduction]
**Learning:** Avoiding implicit `.numpy()` conversion before reduction operations (like MIP) yielded a 16x speedup (0.03s vs 0.6s). Returning Tensors from heavy computation modules allows consumers to chain TF operations efficiently.
**Action:** Prefer returning `tf.Tensor` from TF-heavy classes and handle both Tensor/Numpy in utility functions.

## 2026-01-20 - [API Evolution]
**Learning:** Optimizing return types (e.g. returning Tensors instead of Numpy arrays) can break existing consumers.
**Action:** Use optional arguments (e.g. `return_tensor=False`) to opt-in to optimized return types, preserving backward compatibility while enabling performance improvements.
