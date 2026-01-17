## 2026-01-11 - CLI Loading States for Model Downloads
**Learning:** Users perceive CLI scripts as "frozen" when large model downloads happen silently in the background (e.g., via TensorFlow Hub).
**Action:** Always print a "Loading..." status message before initializing model classes and wrap the initialization in a try/except block to catch network errors gracefully.

## 2026-01-13 - Rich CLI Feedback
**Learning:** Standard print statements get lost in the noise. Using structured output (Panels) and visual indicators (spinners, emojis) significantly improves perceived performance and clarity of CLI tools.
**Action:** Use the `rich` library for all user-facing CLI scripts to provide consistent, accessible, and visually appealing feedback (spinners for blocking ops, panels for context).
