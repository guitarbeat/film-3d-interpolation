# Palette's Journal

## 2024-05-22 - CLI Visual Feedback & Comparative Visualization
**Learning:** In CLI-based data processing tools, "Developer Experience" serves as the User Experience. Users verify results more effectively when scripts output side-by-side comparisons (Start/Interpolated/End) rather than single result images, avoiding the need to manually open multiple files. Additionally, replacing verbose TensorFlow logs with friendly, emoji-based status messages transforms a "debugging" feel into a polished "product" feel.
**Action:** When working on data science/ML scripts, always suppress non-essential library logs (like `TF_CPP_MIN_LOG_LEVEL`) and structure visualization outputs to tell a complete story (Input -> Process -> Output) in a single view.
