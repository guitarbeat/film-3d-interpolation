import os
import warnings

# Suppress TensorFlow logging and warnings for a cleaner CLI experience
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
    print("🎨 Palette: Starting 3D Volume Interpolation Example")

    # Initialize model
    interpolator_3d = Interpolator3D()

    print("🎲 Creating dummy 3D data (Start & End volumes)...")
    # Use different seeds to ensure the volumes are different, making interpolation meaningful.
    volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=1234)
    volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=5678)

    dt = np.array([0.5], dtype=np.float32)

    print("⏳ Interpolating 3D volumes (this may take a moment)...")
    interpolated_volume = interpolator_3d(volume1, volume2, dt)
    print(f"✅ Interpolation complete. Shape: {interpolated_volume.shape}")

    print("📊 Performing Maximum Intensity Projection (MIP) for visualization...")
    # Calculate MIPs for all three states: Start, Interpolated, End
    mip_start = max_intensity_projection(volume1, axis=1)
    mip_interp = max_intensity_projection(interpolated_volume, axis=1)
    mip_end = max_intensity_projection(volume2, axis=1)

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'interpolated_mip.png')

    # Create a side-by-side comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Start Frame
    axes[0].imshow(mip_start[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Start (t=0)")
    axes[0].axis('off')

    # Plot Interpolated Frame
    # Note: Interpolator returns 3 channels (RGB), we take the first one for grayscale viz
    axes[1].imshow(mip_interp[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Interpolated (t=0.5)")
    axes[1].axis('off')

    # Plot End Frame
    axes[2].imshow(mip_end[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("End (t=1)")
    axes[2].axis('off')

    plt.suptitle("3D Volume Interpolation: Maximum Intensity Projections", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"🖼️  Saved comparison visualization to {out_path}")
