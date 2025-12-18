import os

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    print("✨ Starting 3D Interpolation Demo ✨")

    print("🎨 Loading FILM model from TF Hub (this may take a moment)...")
    interpolator_3d = Interpolator3D()

    print("🎲 Creating dummy 3D data...")
    # Use different seeds to ensure the volumes are different, making interpolation meaningful.
    volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=1234)
    volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5, seed=5678)

    dt = np.array([0.5], dtype=np.float32)

    print("🔄 Interpolating 3D volumes...")
    interpolated_volume = interpolator_3d(volume1, volume2, dt)
    print(f"✅ Interpolation complete. Shape: {interpolated_volume.shape}")

    print("📊 Performing Maximum Intensity Projection (MIP) for comparison...")
    # Calculate MIPs for all three volumes for comparison
    mip_v1 = max_intensity_projection(volume1, axis=1)
    mip_interp = max_intensity_projection(interpolated_volume, axis=1)
    mip_v2 = max_intensity_projection(volume2, axis=1)

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'interpolated_comparison_mip.png')

    # Create a comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Helper to plot
    def plot_mip(ax, data, title):
        # Handle 1-channel data for display
        if data.shape[-1] == 1:
            display_data = data[0, :, :, 0]
            cmap = 'gray'
        else:
            display_data = data[0]
            cmap = None

        im = ax.imshow(display_data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        return im

    plot_mip(axes[0], mip_v1, "Start Volume (t=0)")
    plot_mip(axes[1], mip_interp, "Interpolated (t=0.5)")
    im = plot_mip(axes[2], mip_v2, "End Volume (t=1)")

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Intensity")

    plt.suptitle("3D Volume Interpolation - Maximum Intensity Projections", fontsize=16)
    plt.savefig(out_path, bbox_inches='tight')

    print(f"🖼️  Saved comparison MIP image to: {out_path}")
    print("👋 Demo finished successfully!")
