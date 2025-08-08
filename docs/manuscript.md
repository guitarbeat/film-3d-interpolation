## FILM-3D: Slice-wise Frame Interpolation for 3D Volumetric Data and Time

### Authors
- [Your Name], [Affiliation]
- [Co-author], [Affiliation]

### Abstract
We present a pragmatic extension of Frame Interpolation for Large Motion (FILM) to higher-dimensional data by adapting a pre-trained 2D FILM model to operate slice-wise on 3D volumes across time. Our method processes each axial slice with the 2D FILM interpolator and reassembles the output into a 3D volume, enabling temporal interpolation for volumetric data without retraining. We further provide a simple visualization pipeline via Maximum Intensity Projection (MIP) to summarize interpolated volumes. Using synthetic “stick” phantoms, we demonstrate end-to-end functionality and discuss limitations and paths toward a native 3D solution.

### 1. Introduction
Temporal interpolation for volumetric imaging (e.g., MRI, CT, microscopy) can reduce acquisition time, improve temporal resolution, and enable motion-aware analysis. While modern interpolation models such as FILM offer high-quality results for 2D natural images, extending them to 3D+time remains challenging due to memory constraints, anisotropy, and model design.

This work provides an initial, engineering-focused step: a slice-wise adaptation that leverages a pre-trained 2D FILM model from TensorFlow Hub to interpolate between two 3D volumes at adjacent time points. We complement this with an efficient MIP-based visualization to condense the interpolated volume into a 2D summary.

### 2. Related Work
- FILM (Frame Interpolation for Large Motion): High-quality 2D frame interpolation leveraging large motion handling and efficient architectures. We build directly atop the public TensorFlow Hub implementation.
- Volumetric reconstruction and interpolation: Prior work in medical imaging and 3D microscopy explores learned 3D interpolation and super-resolution; our focus here is a minimal viable path to 3D+time using existing 2D models.
- Maximum Intensity Projection (MIP): A standard technique to visualize volumetric data by projecting the brightest voxels along a chosen axis.

### 3. Method
#### 3.1 Data representation
We represent inputs as 5D tensors with shape \(B \times D \times H \times W \times C\) where \(B\) is batch size, \(D\) depth (slices), \(H\) height, \(W\) width, and \(C\) channels.

#### 3.2 Preprocessing and alignment
To satisfy model alignment constraints, we pad 2D slices so \(H\) and \(W\) are divisible by an alignment factor (default 64), then crop outputs back to the original size. See `film_3d.py` function `_pad_to_align`.

#### 3.3 Slice-wise FILM inference
Given two 3D volumes `x0` and `x1` and an interpolation time `dt ∈ [0, 1]`, we iterate over depth slices `d ∈ {1..D}` and apply the 2D FILM model to the pair of 2D images `x0[:, d, ...]` and `x1[:, d, ...]`. For single-channel inputs, we repeat channels to meet the 3-channel requirement. The per-slice outputs are stacked along depth to form the interpolated volume. See class `Interpolator3D` in `film_3d.py`.

#### 3.4 Maximum Intensity Projection (MIP)
For visualization we compute MIP along the depth axis to obtain a 2D image per volume: `max_intensity_projection(volume, axis=1)` in `film_3d.py`.

### 4. Implementation Details
- Model: TensorFlow Hub FILM (`https://tfhub.dev/google/film/1`), loaded in `Interpolator3D`.
- Alignment: Default `align=64` for height/width divisibility.
- Channel handling: 1-channel volumes are expanded to 3 channels via repetition.
- Dependencies: see `requirements.txt` (TensorFlow 2.x, TensorFlow Hub, NumPy, Matplotlib, etc.).
- Repository structure: `film_3d.py` (core), `run_mip_example.py` (demo), `test_film_3d.py` (unit tests), `README.md` (usage), `interpolated_mip.png` (example output).

### 5. Experiments
#### 5.1 Synthetic “stick” phantoms
We generate 3D volumes containing linear high-intensity structures (“sticks”) with random orientation and position using `create_dummy_3d_data` in `run_mip_example.py`. Two independent volumes serve as endpoints for interpolation with `dt=0.5`.

#### 5.2 Protocol
- Construct `volume1`, `volume2` with shape `(1, 10, 64, 64, 1)`.
- Interpolate with `Interpolator3D(volume1, volume2, dt=[0.5])`.
- Visualize via MIP along depth; save to `interpolated_mip.png`.

#### 5.3 Qualitative results
The produced MIP highlights the brightest voxels in the interpolated volume, qualitatively showing coherent structures derived from the endpoints.

![Interpolated MIP](../examples/outputs/interpolated_mip.png)

### 6. Validation
Unit tests in `test_film_3d.py` validate:
- `load_volume`: returns a normalized dummy 3D array with expected shape and dtype.
- `max_intensity_projection`: produces correct projections on controlled inputs.

### 7. Discussion
#### Strengths
- Zero-training path to interpolate 3D+time by reusing a strong 2D model.
- Simple, modular design; minimal changes needed to plug in real data loaders.
- Deterministic spatial alignment via padding/cropping.

#### Limitations
- No explicit 3D reasoning; slice-wise processing may introduce inter-slice inconsistencies.
- Assumes near-isotropic in-plane resolution; anisotropy and thick slices can degrade quality.
- Single-channel volumes are naively triplicated; modality-specific encoders could be preferable.
- Current demo is qualitative; no quantitative benchmarks yet.

#### Ethical and domain considerations
When applied to clinical data, interpolation artifacts can mislead interpretation. Rigorous validation and uncertainty estimation are required before deployment.

### 8. Future Work
- Native 3D/2.5D modeling: 3D kernels or multi-slice context for cross-slice consistency.
- Training and datasets: synthetic-to-real curricula; domain-specific augmentations.
- Metrics: 3D PSNR/SSIM, perceptual metrics (e.g., LPIPS variants) on volumes.
- Temporal consistency: enforce smoothness across time and depth.
- Uncertainty: probabilistic outputs or ensemble strategies for reliability.

### 9. Reproducibility
#### Environment
```bash
pip install -r requirements.txt
```

#### Run demo
```bash
PYTHONPATH=src python3 examples/run_mip_example.py
```
This generates `examples/outputs/interpolated_mip.png`.

#### Notes
- For reproducible dummy data, an RNG seed is set in `examples/run_mip_example.py`.
- The demo downloads the FILM model at first run; ensure network access.

### 10. Conclusion
We provide a compact, working pathway to apply state-of-the-art 2D frame interpolation to volumetric data and time by operating slice-wise, plus a lightweight MIP visualization. This forms a baseline for more principled 3D models and datasets.

### References
- TensorFlow Hub FILM: [google/film/1](https://tfhub.dev/google/film/1)
- Maximum Intensity Projection (MIP): standard technique in volume visualization (e.g., medical imaging texts).