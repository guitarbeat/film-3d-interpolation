## 2026-01-02 - Side-by-Side Visual Verification in CLI Tools
**Learning:** Users running scientific scripts (like interpolation) often lack context when only the *result* is shown. Providing a side-by-side comparison (Start -> Result -> End) immediately validates the algorithm's behavior and builds trust.
**Action:** Always structure visualization outputs in examples/scripts to show the input state alongside the output state for direct comparison.

## 2026-01-02 - Polished CLI Output
**Learning:** Raw library logs (like TensorFlow info/warnings) clutter the output and make the tool feel "broken" or "noisy". Suppressing them and replacing them with clean, emoji-guided status messages dramatically improves the perceived quality of the tool.
**Action:** Use `os.environ` and `warnings` filters to silence backend noise, and implement friendly print statements with `sys.stdout.flush()` for immediate feedback.
