## 2026-01-11 - CLI Loading States for Model Downloads
**Learning:** Users perceive CLI scripts as "frozen" when large model downloads happen silently in the background (e.g., via TensorFlow Hub).
**Action:** Always print a "Loading..." status message before initializing model classes and wrap the initialization in a try/except block to catch network errors gracefully.

## 2026-01-17 - Device Reporting & Timing Feedback
**Learning:** For compute-intensive tasks, users need to know the execution context (CPU vs GPU) and performance metrics (timing) to manage expectations and verify hardware utilization.
**Action:** Always report the active device at startup and provide execution time for long-running operations in CLI scripts.
