## 2026-01-11 - CLI Loading States for Model Downloads
**Learning:** Users perceive CLI scripts as "frozen" when large model downloads happen silently in the background (e.g., via TensorFlow Hub).
**Action:** Always print a "Loading..." status message before initializing model classes and wrap the initialization in a try/except block to catch network errors gracefully.

## 2026-01-13 - Rich CLI Feedback
**Learning:** Standard print statements get lost in the noise. Using structured output (Panels) and visual indicators (spinners, emojis) significantly improves perceived performance and clarity of CLI tools.
**Action:** Use the `rich` library for all user-facing CLI scripts to provide consistent, accessible, and visually appealing feedback (spinners for blocking ops, panels for context).

## 2026-01-18 - Scientific CLI Structure
**Learning:** For data-intensive CLI tools, users value explicit step tracking ("[1/3]") and a final data summary table (input vs output shapes) to verify correctness without needing to open output files immediately.
**Action:** Implement step counters in status messages and a 'Data Summary' table at the end of processing scripts to confirm dimensional integrity.

## 2026-01-25 - Robust Status Fallbacks
**Learning:** When `rich` is missing, simple print statements ("Loading...") leave users unsure if a process has finished or hung.
**Action:** Implement a lightweight context manager for non-`rich` environments that appends "Done." or "Failed." upon completion, ensuring the user always receives closure for blocking operations.
