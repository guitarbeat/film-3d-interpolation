## 2026-01-03 - Improved CLI DX and Visualization Context
**Learning:** For ML example scripts, users need context (input vs output) to verify results, and friendly status messages (with emojis/flushing) prevent the "is it hung?" anxiety during heavy model loads.
**Action:** When creating CLI tools, suppress verbose library logs, flush stdout immediately on status updates, and prefer side-by-side "before/after" visualizations over single result images.
