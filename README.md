# FILM for 3D Data and Time

This repository aims to extend the Frame Interpolation for Large Motion (FILM) model to handle higher-dimensional data, specifically 3D volumetric data over time. The initial focus is on implementing maximum intensity projections for 3D sticks in time.

## Project Plan

### Phase 1: Data Representation and Preprocessing

- Define a suitable data structure for 3D volumetric data with a time component.
- Implement functions for loading and preprocessing 3D time-series data.
- Consider normalization and scaling for optimal model performance.

### Phase 2: Adapting FILM for 3D

- Investigate how the FILM architecture can be modified to accept 3D input.
- Explore techniques for handling the temporal dimension (e.g., treating time as an additional spatial dimension or using recurrent layers).
- Implement the core 3D interpolation logic.

### Phase 3: Maximum Intensity Projection (MIP)

- Implement a function to perform Maximum Intensity Projection along a specified axis (e.g., Z-axis for 3D sticks).
- Integrate MIP into the interpolation pipeline to visualize 2D projections of the 3D interpolated data.

### Phase 4: Training and Evaluation

- Develop a strategy for generating synthetic 3D time-series data for training and testing.
- Define appropriate loss functions and evaluation metrics for 3D interpolation.
- Train the adapted FILM model and evaluate its performance.

### Phase 5: Application to 3D Sticks in Time

- Apply the developed model to interpolate 3D stick data over time.
- Visualize the interpolated 3D sticks using MIP.

## Usage

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Maximum Intensity Projection (MIP) example:**

    ```bash
    PYTHONPATH=src python3 examples/run_mip_example.py
    ```

    This will generate an image at `examples/outputs/interpolated_mip.png` showing the 2D MIP of the interpolated 3D data.

## Code Structure

- `src/film_3d/`: Contains the core implementation:
  - `__init__.py`: `Interpolator3D`, `max_intensity_projection`, and utility functions.
- `examples/`: Example scripts and their outputs
  - `run_mip_example.py`: Demonstrates interpolation and MIP visualization.
  - `outputs/`: Generated figures and artifacts (gitignored).
- `tests/`: Unit tests
  - `test_film_3d.py`: Tests for `load_volume` and `max_intensity_projection`.
- `docs/`: Documentation and manuscript drafts
  - `manuscript.md`: Draft manuscript in Markdown.

## Notes
- Ensure network access on first run to download the FILM model from TensorFlow Hub.
- For deterministic synthetic data, set or adjust the RNG seed in `examples/run_mip_example.py`.


