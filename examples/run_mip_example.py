import os
import sys
import warnings

# Suppress TensorFlow logs and warnings for a cleaner CLI experience
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from film_3d import Interpolator3D, max_intensity_projection


def create_dummy_3d_data(shape: tuple = (1, 10, 64, 64, 1), num_sticks: int = 5, stick_length: int = 5, seed: int = 1234) -> np.ndarray:
    """Creates dummy 3D volumetric data containing simple 'sticks' for demonstration purposes.

    This function generates a 5D NumPy array representing a batch of 3D volumes.
    Each volume contains a specified number of randomly placed 'sticks' (lines of high intensity).
    This is useful for testing 3D interpolation and MIP functions without needing real data.
    """
    data = np.zeros(shape, dtype=np.float32)
    batch_size, depth, height, width, channels = shape

    rng = np.random.default_rng(seed)

    for b in range(batch_size):
        for _ in range(num_sticks):
            start_d = rng.integers(0, depth - stick_length + 1)
            start_h = rng.integers(0, height)
            start_w = rng.integers(0, width)
            orientation = rng.choice([0, 1, 2])  # 0: depth, 1: height, 2: width
            for i in range(stick_length):
                if orientation == 0 and start_d + i < depth:
                    data[b, start_d + i, start_h, start_w, 0] = 1.0
                elif orientation == 1 and start_h + i < height:
                    data[b, start_d, start_h + i, start_w, 0] = 1.0
                elif orientation == 2 and start_w + i < width:
                    data[b, start_d, start_h, start_w + i, 0] = 1.0
    return data


if __name__ == '__main__':
    print("🎨 Palette: Initializing 3D Interpolation Example...", flush=True)

    interpolator_3d = Interpolator3D()

    print("🎲 Creating dummy 3D data...", flush=True)
    # Use different seeds to ensure the volumes are different, making interpolation meaningful.
    volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=1234)
    volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=5678)

    dt = np.array([0.5], dtype=np.float32)

    print("⏳ Interpolating 3D volumes...", flush=True)
    interpolated_volume = interpolator_3d(volume1, volume2, dt)
    print("✅ Interpolation complete. Interpolated volume shape:", interpolated_volume.shape, flush=True)

    print("📊 Performing Maximum Intensity Projection...", flush=True)
    mip_start = max_intensity_projection(volume1, axis=1)
    mip_interp = max_intensity_projection(interpolated_volume, axis=1)
    mip_end = max_intensity_projection(volume2, axis=1)
    print("MIP image shape:", mip_interp.shape, flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'interpolated_mip.png')

    # Create a side-by-side comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Start Volume MIP
    axes[0].imshow(mip_start[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Start Volume (t=0)")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")

    # Interpolated Volume MIP
    # Note: Interpolated volume is 3-channel (RGB), so we don't use cmap='gray' if we want to see colors,
    # but for consistency with input which is grayscale, we might want to just show the first channel
    # or keep it as RGB if the model introduced color artifacts/features.
    # The memory says "FILM model outputs 3-channel (RGB)... visualization code must handle this".
    # Since inputs are grayscale sticks, RGB output might just be grayscale values in 3 channels.
    # Clip values to [0, 1] to avoid matplotlib warnings about range.
    axes[1].imshow(np.clip(mip_interp[0, :, :, :], 0, 1))
    axes[1].set_title("Interpolated Volume (t=0.5)")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")

    # End Volume MIP
    axes[2].imshow(mip_end[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("End Volume (t=1)")
    axes[2].set_xlabel("Width")
    axes[2].set_ylabel("Height")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"🖼️  Saved comparison MIP image to {out_path}", flush=True)
