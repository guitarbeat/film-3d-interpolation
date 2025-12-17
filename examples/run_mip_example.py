import os
# Suppress TensorFlow logs for cleaner CLI output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run FILM 3D MIP Example")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), 'outputs'),
                        help="Directory to save outputs")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Number of frames to interpolate between 0 and 1. If > 1, multiple frames are generated.")
    parser.add_argument("--depth", type=int, default=10, help="Depth of the volume")
    parser.add_argument("--height", type=int, default=64, help="Height of the volume")
    parser.add_argument("--width", type=int, default=64, help="Width of the volume")
    parser.add_argument("--num-sticks", type=int, default=5, help="Number of sticks")
    parser.add_argument("--stick-length", type=int, default=5, help="Length of sticks")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for volume 1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("Initializing Interpolator3D (loading model)...")
    interpolator_3d = Interpolator3D()

    shape = (1, args.depth, args.height, args.width, 1)

    print(f"Creating dummy 3D data with shape {shape}...")
    # Use different seeds to ensure the volumes are different, making interpolation meaningful.
    volume1 = create_dummy_3d_data(shape=shape, num_sticks=args.num_sticks, stick_length=args.stick_length, seed=args.seed)
    volume2 = create_dummy_3d_data(shape=shape, num_sticks=args.num_sticks, stick_length=args.stick_length, seed=args.seed + 1234)

    # Determine time points
    if args.num_frames == 1:
        time_points = [0.5]
    else:
        # Generate N frames evenly spaced between 0 and 1 (exclusive)
        # e.g. 3 frames -> 0.25, 0.5, 0.75
        time_points = np.linspace(0, 1, args.num_frames + 2)[1:-1]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Interpolating {len(time_points)} frames...")

    # Iterate over time points with a progress bar
    for i, t in enumerate(tqdm(time_points, desc="Interpolating")):
        dt = np.array([t], dtype=np.float32)
        interpolated_volume = interpolator_3d(volume1, volume2, dt)

        mip_image = max_intensity_projection(interpolated_volume, axis=1)

        # Save image
        out_filename = f'interpolated_mip_t{t:.2f}.png'
        if args.num_frames == 1:
             out_filename = 'interpolated_mip.png'

        out_path = os.path.join(args.output_dir, out_filename)

        plt.figure(figsize=(6, 6))
        plt.imshow(mip_image[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.title(f"MIP (t={t:.2f})")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        # Only add colorbar once or it messes up layout in loop?
        # Actually standard plt usage creates new figure each time if we don't close.
        # But we called plt.figure() inside loop.
        cbar = plt.colorbar()
        cbar.set_label("Intensity")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close() # Close to free memory

    print(f"Saved {len(time_points)} MIP images to {args.output_dir}")
