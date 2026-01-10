# Palette's Journal

This document tracks critical UX and accessibility learnings for this project.

## 2024-05-22 - CLI Async Feedback
**Learning:** Users perceive "silent" CLI operations (like large model downloads) as hung processes.
**Action:** Always wrap heavy initialization steps with a "Loading..." status message and try/except blocks for friendly error handling.
