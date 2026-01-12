## 2024-05-23 - [Handling Network Errors in Library Initialization]
**Learning:** Initializing deep learning models often involves implicit network calls (e.g., `hub.load`). If these fail (offline, timeout), standard tracebacks are confusing for users who might not realize a download is happening.
**Action:** Always wrap these implicit network calls in `try/except` blocks to provide immediate, actionable feedback (e.g., "Check internet connection") and a cleaner error message.
