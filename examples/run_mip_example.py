import os
import warnings
import contextlib
import sys

# Suppress TensorFlow logs and warnings for a cleaner CLI experience
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from film_3d import Interpolator3D, max_intensity_projection

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

@contextlib.contextmanager
def cli_status(msg):
    if RICH_AVAILABLE:
        with console.status(f"[bold green]{msg}", spinner="dots"):
            yield
    else:
        print(f"⏳ {msg}...", flush=True)
        yield
        print("✅ Done.", flush=True)

def create_dummy_3d_data(shape: tuple = (1, 10, 64, 64, 1), num_sticks: int = 5, stick_length: int = 5, seed: int = 1234) -> np.ndarray:
    """Creates dummy 3D volumetric data containing simple 'sticks' for demonstration purposes."""
    data = np.zeros(shape, dtype=np.float32)
    batch_size, depth, height, width, channels = shape
    rng = np.random.default_rng(seed)
    for b in range(batch_size):
        for _ in range(num_sticks):
            start_d = rng.integers(0, depth - stick_length + 1)
            start_h = rng.integers(0, height)
            start_w = rng.integers(0, width)
            orientation = rng.choice([0, 1, 2])
            for i in range(stick_length):
                if orientation == 0 and start_d + i < depth:
                    data[b, start_d + i, start_h, start_w, 0] = 1.0
                elif orientation == 1 and start_h + i < height:
                    data[b, start_d, start_h + i, start_w, 0] = 1.0
                elif orientation == 2 and start_w + i < width:
                    data[b, start_d, start_h, start_w + i, 0] = 1.0
    return data

if __name__ == '__main__':
    # Detect device
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        device_msg = f"🚀 Accelerated on {len(gpus)} GPU(s)" if gpus else "🐢 Running on CPU"
    except:
        device_msg = "❓ Unknown device"

    if RICH_AVAILABLE:
        console.print(Panel(f"[bold blue]FILM 3D Interpolation[/bold blue]\n{device_msg}", expand=False))
    else:
        print(f"--- FILM 3D Interpolation ---\n{device_msg}\n-----------------------------")

    try:
        with cli_status("Loading FILM model"):
            interpolator_3d = Interpolator3D()
    except Exception as e:
        msg = f"❌ Error loading FILM model: {e}\n💡 Check internet connection."
        if RICH_AVAILABLE: console.print(f"[bold red]{msg}[/bold red]")
        else: print(msg)
        exit(1)

    with cli_status("Creating dummy 3D data"):
        volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), seed=1234)
        volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), seed=5678)
        dt = np.array([0.5], dtype=np.float32)

    with cli_status("Interpolating 3D volumes"):
        interpolated_volume = interpolator_3d(volume1, volume2, dt)

    with cli_status("Performing MIP"):
        mip_v1 = max_intensity_projection(volume1, axis=1)
        mip_interp = max_intensity_projection(interpolated_volume, axis=1)
        mip_v2 = max_intensity_projection(volume2, axis=1)

    out_path = os.path.join(os.path.dirname(__file__), 'outputs', 'interpolated_mip_comparison.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with cli_status(f"Saving to {out_path}"):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(np.clip(mip_v1[0, :, :, 0], 0, 1), cmap='gray'); plt.title("Start"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(np.clip(mip_interp[0], 0, 1)); plt.title("Interpolated"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(np.clip(mip_v2[0, :, :, 0], 0, 1), cmap='gray'); plt.title("End"); plt.axis('off')
        plt.tight_layout(); plt.savefig(out_path)

    msg = f"✨ Done! Check output at {out_path}"
    if RICH_AVAILABLE: console.print(f"[bold green]{msg}[/bold green]")
    else: print(msg)
