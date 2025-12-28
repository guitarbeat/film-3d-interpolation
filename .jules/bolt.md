## 2024-05-23 - [TensorFlow Channel Expansion]
**Learning:** `tf.image.grayscale_to_rgb` is significantly faster (>30%) than `tf.repeat(x, 3, axis=-1)` for expanding single-channel inputs to 3 channels.
**Action:** Prefer `tf.image.grayscale_to_rgb` when converting grayscale images to RGB for model input.
