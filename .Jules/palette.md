## 2024-05-23 - Visual Verification in CLI Tools
**Learning:** For CLI tools generating image/volume data, users struggle to verify results without side-by-side comparison of input vs output.
**Action:** Always generate 3-panel visualizations (Start, Result, End) for interpolation/processing tasks instead of just the result.

## 2024-05-23 - Blocking Operation Feedback
**Learning:** Model loading in scripts often looks like a hang.
**Action:** Always print a status message (with emoji) *before* the blocking call, not just after.
