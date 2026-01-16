## 2026-01-16 - Graph Execution for Mixed Inputs
**Learning:** In `Interpolator3D`, encapsulating the model call and immediate preprocessing (like `grayscale_to_rgb`) within a method decorated with `@tf.function` reduced overhead by ~2.5% compared to Eager execution, especially when inputs are mixed (NumPy/Tensor).
**Action:** Use `@tf.function` to wrap atomic inference steps that involve tensor ops, even if the model itself is already a SavedModel.
