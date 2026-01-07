## 2026-01-07 - CLI & Visualization UX
**Learning:** In headless/data-science environments, "UX" often means providing clear, immediate CLI feedback (using `flush=True` and emojis for status) and generating self-explanatory artifacts (side-by-side comparisons instead of single images) so users don't have to manually inspect raw data.
**Action:** When updating example scripts, always suppress noisy library logs (like TF warnings) and format the final output image to tell a complete story (e.g., Before -> After).
