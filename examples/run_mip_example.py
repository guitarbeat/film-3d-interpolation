import os
import warnings

# Suppress TensorFlow logs and warnings for cleaner CLI output
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
    # Initialize interpolator
    interpolator_3d = Interpolator3D()

    print("Creating dummy 3D data... 🧊", flush=True)
    # Use different seeds to ensure the volumes are different, making interpolation meaningful.
    volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=1234)
    volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=5678)

    dt = np.array([0.5], dtype=np.float32)

    print("Interpolating 3D volumes... 🔄", flush=True)
    interpolated_volume = interpolator_3d(volume1, volume2, dt)
    print("Interpolation complete. ✅", flush=True)
    print(f"  - Interpolated volume shape: {interpolated_volume.shape}")

    print("Performing Maximum Intensity Projection (MIP)... 📐", flush=True)
    # Compute MIP for start, interpolated, and end volumes
    mip_v1 = max_intensity_projection(volume1, axis=1)
    mip_interp = max_intensity_projection(interpolated_volume, axis=1)
    mip_v2 = max_intensity_projection(volume2, axis=1)

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'interpolated_mip_comparison.png')

    print("Generating side-by-side comparison... 🎨", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Helper to plot MIP
    def plot_mip(ax, data, title):
        # Handle 3-channel output from interpolator by taking the first channel if it's greyscale-like
        # or clipping if it's RGB. Here we assume we want to visualize intensity.
        # Since input was 1-channel, let's visualize the 0th channel for consistency.
        if data.shape[-1] == 3:
            img_data = data[0, :, :, 0]
        else:
            img_data = data[0, :, :, 0]

        im = ax.imshow(img_data, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        return im

    plot_mip(axes[0], mip_v1, "Start Volume (t=0)")
    plot_mip(axes[1], mip_interp, "Interpolated (t=0.5)")
    plot_mip(axes[2], mip_v2, "End Volume (t=1)")

    # Add a common colorbar
    cbar = fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Intensity")

    plt.suptitle("3D Volume Interpolation - MIP Visualization", fontsize=14)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison MIP image to {out_path} 💾", flush=True)
