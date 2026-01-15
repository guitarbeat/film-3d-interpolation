## 2026-01-11 - CLI Loading States for Model Downloads
**Learning:** Users perceive CLI scripts as "frozen" when large model downloads happen silently in the background (e.g., via TensorFlow Hub).
**Action:** Always print a "Loading..." status message before initializing model classes and wrap the initialization in a try/except block to catch network errors gracefully.

## 2026-01-15 - Transitive Dependencies for UX
**Learning:** `rich` is often available via `tensorflow` or `keras` even if not explicitly installed in `requirements.txt`.
**Action:** Check for `rich` availability (try/except import) and use it for improved CLI UX (spinners, panels) without forcing new dependencies, but always implement a robust standard library fallback.
