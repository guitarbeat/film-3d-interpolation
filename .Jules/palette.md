## 2024-05-23 - CLI UX for Scientific Scripts
**Learning:** Scientists/Researchers often run long blocking scripts (model loading, inference). Standard CLI output is often verbose with framework logs (TF/PyTorch) and sparse on progress.
**Action:** Always suppress framework logs by default in example scripts (`TF_CPP_MIN_LOG_LEVEL='2'`). Add "delight" through clear emoji-based status indicators for blocking steps. Use side-by-side visualization for output verification instead of single result files.
