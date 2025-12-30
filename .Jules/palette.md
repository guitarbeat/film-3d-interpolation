## 2024-05-23 - CLI Developer Experience
**Learning:** For CLI-based tools without a GUI, the "User Interface" is the terminal output. Suppressing verbose framework logs (like TensorFlow) and using emojis/clear status messages *before* blocking operations significantly improves perceived performance and friendliness.
**Action:** Always filter framework warnings and use `sys.stdout.flush()` after printing "loading" messages to ensure they appear before the heavy import or operation starts.

## 2024-05-23 - Contextual Visualization
**Learning:** Single-image outputs for interpolation or transformation tasks are often insufficient for users to judge quality. Side-by-side comparisons (Start -> Result -> End) provide necessary context.
**Action:** When generating visual artifacts for examples, always produce composite images that show inputs alongside outputs.
