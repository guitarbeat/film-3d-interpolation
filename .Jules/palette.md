## 2026-01-11 - CLI Loading States for Model Downloads
**Learning:** Users perceive CLI scripts as "frozen" when large model downloads happen silently in the background (e.g., via TensorFlow Hub).
**Action:** Always print a "Loading..." status message before initializing model classes and wrap the initialization in a try/except block to catch network errors gracefully.

## 2026-01-14 - Progressive Enhancement for CLI UX
**Learning:** Adding optional dependencies like `rich` allows for a premium developer experience (spinners, panels) without complicating the setup for users who want minimal dependencies.
**Action:** Use `try/except ImportError` to conditionally import UX libraries and provide robust standard library fallbacks (e.g., plain print vs rich status).
