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

## 2026-02-03 - Scientific Data Summaries
**Learning:** Users working with normalized scientific data (e.g., [0, 1] range) need immediate validation of value ranges and data types in CLI outputs to catch subtle errors (like overflow or incorrect scaling) without opening external tools.
**Action:** Enhance data summary tables to include `dtype` and `Range (Min/Max)` columns, alongside shape information, for all tensor/array outputs.
## 2026-01-26 - Data Transparency in CLI Summaries
**Learning:** In scientific workflows, verifying tensor shapes is insufficient; silent type conversions or normalization issues (e.g., float vs uint8 ranges) are common pitfalls.
**Action:** Enhance CLI data summaries to include `dtype` and `min/max` range statistics alongside shapes, giving users immediate confidence in data integrity.

## 2026-02-12 - Persistent Completion Messages in CLI
**Learning:** Users lose context in multi-step CLI processes when transient status messages (spinners) disappear without a permanent completion record, creating a disjointed log.
**Action:** Design status context managers to automatically append a persistent "Success" message upon completion, ensuring a linear and complete history of execution steps.
