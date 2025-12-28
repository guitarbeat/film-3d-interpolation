## 2025-12-28 - 3D Volume Comparison Visualization
**Learning:** Users running ML interpolation examples need to verify the output quality by comparing it to the inputs. A single output image is insufficient context.
**Action:** Always generate side-by-side comparison plots (Start | Interpolated | End) for ML generative tasks.

## 2025-12-28 - CLI "Hanging" Perception
**Learning:** Loading large ML models (like FILM from TF Hub) can pause execution for several seconds without feedback, making the user think the script has crashed.
**Action:** Print a friendly status message (e.g., "⏳ Interpolating...") *immediately before* the blocking call, not after.

## 2025-12-28 - TensorFlow Noise
**Learning:** TensorFlow's default logging is extremely verbose and scary for end-users (warnings about CUDA, optimizations, etc.), detracting from the "delightful" experience.
**Action:** In high-level example scripts intended for users, suppress these logs using `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'` and `warnings.filterwarnings('ignore')`.
