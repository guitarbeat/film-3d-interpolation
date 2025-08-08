## Roadmap and Task List

### Status
- Repo structure: done
- Demo + unit tests: done
- Manuscript draft: done

### Guiding Goals
- **Short-term**: Robust slice-wise baseline with real data IO, metrics, and CI.
- **Mid-term**: Add multi-slice context (2.5D), better visualization, packaging/CLI.
- **Long-term**: Native 3D model and training pipeline with quantitative benchmarks.

### Phase 1 — Data IO and Preprocessing (P0)
- [ ] Implement real 3D volume loaders
  - [ ] NIfTI (`.nii/.nii.gz`) via `nibabel`
  - [ ] TIFF stacks via `tifffile`
  - [ ] DICOM series via `pydicom` (optional, later if needed)
- [ ] Standardize dtype/range handling (float32 in [0,1])
- [ ] Handle metadata (spacing, anisotropy); store in a `VolumeMeta` struct
- [ ] Deterministic seeding and synthetic data controls (expose in example)

### Phase 2 — Model API and Inference (P0)
- [ ] Add 2.5D context: feed k adjacent slices to 2D FILM (configurable)
- [ ] Batched depth-slice inference; progress reporting with `tqdm`
- [ ] Mixed precision + XLA toggles; VRAM/throughput profiling
- [ ] Robust channel handling (C∈{1,3}; plug-in converters)

### Phase 3 — Evaluation and Metrics (P0)
- [ ] Synthetic dataset generator with param sweeps (density, orientation, SNR)
- [ ] Metrics: 3D PSNR, 3D SSIM; (optional) LPIPS on MIP or slice averages
- [ ] Reproducible experiment scripts under `examples/`
- [ ] Figure utilities: grids of slices, MIPs across axes

### Phase 4 — Visualization and MIP (P1)
- [ ] Generalize MIP to arbitrary axis; add min/mean/percentile projections
- [ ] Window/level, gamma, colormaps; safe defaults
- [ ] Save multi-panel PNGs and simple HTML reports

### Phase 5 — Testing and CI (P0)
- [ ] Unit tests for loaders, preprocessing, projections
- [ ] Mock TF-Hub model for `Interpolator3D` tests (no network, fast)
- [ ] Linting/formatting (ruff/black) and type checking (mypy/pyright)
- [ ] GitHub Actions: tests on push/PR (CPU-only matrix)

### Phase 6 — Packaging and CLI (P1)
- [ ] Convert to installable package (`pyproject.toml`)
- [ ] CLI: `film3d interpolate --x0 ... --x1 ... --dt 0.5 --mip axis=1`
- [ ] Example notebooks under `docs/` or `examples/`

### Phase 7 — Training Path (Exploratory) (P2)
- [ ] Baseline: per-slice fine-tuning of 2D FILM (if feasible)
- [ ] Lightweight 3D or 2.5D model prototype; compare to slice-wise FILM
- [ ] Benchmarks on public volumetric datasets (microscopy/medical where permitted)

### Phase 8 — Manuscript and Docs (P1)
- [ ] Quantitative tables + ablations (context size, alignment, metrics)
- [ ] Failure cases and limitations (anisotropy, inter-slice drift)
- [ ] Ethics/safety note for clinical use

### Next 1–2 Weeks (Suggested Order)
1. [ ] Data loaders (NIfTI, TIFF) + tests — Phase 1
2. [ ] Metric utilities (PSNR/SSIM3D) + synthetic dataset script — Phase 3
3. [ ] CI + mock TF-Hub for fast tests — Phase 5
4. [ ] Generalized MIP/projections and fig utils — Phase 4

### Open Questions (for planning)
- Which modalities/datasets are priority (microscopy vs. medical)?
- Required projections/visualizations for your use case?
- Is 2.5D context sufficient short-term, or jump straight to 3D model?
- Packaging target: internal tool or public PyPI package?
- Constraints on runtime (GPU/CPU), memory, and environment?