import os
import warnings
import time
import contextlib

# Suppress TensorFlow logs and warnings for a cleaner CLI experience
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from film_3d import Interpolator3D, max_intensity_projection

# Try to import rich for better UX, otherwise fall back to standard print
try:
    from rich.console import Console
    from rich.traceback import install
    from rich.panel import Panel
    from rich.table import Table
    install()
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# UX Helpers
def print_header(title):
    if HAS_RICH:
        console.print(Panel.fit(f"[bold blue]{title}[/bold blue]", border_style="blue"))
    else:
        print(f"\n=== {title} ===\n")

def print_status(msg, spinner="dots"):
    if HAS_RICH:
        return console.status(f"[bold]{msg}[/bold]", spinner=spinner)

    @contextlib.contextmanager
    def simple_status():
        print(msg, end=" ", flush=True)
        try:
            yield
            print("Done.")
        except:
            print("Failed.")
            raise

    return simple_status()

def print_success(msg):
    if HAS_RICH:
        console.print(f"[bold green]✅ {msg}[/bold green]")
    else:
        print(f"✅ {msg}")

def print_error(msg):
    if HAS_RICH:
        console.print(f"[bold red]❌ {msg}[/bold red]")
    else:
        print(f"❌ {msg}")

def print_summary(v1, v2, interp):
    def get_info(arr):
        if tf.is_tensor(arr):
            val_min = float(tf.reduce_min(arr))
            val_max = float(tf.reduce_max(arr))
            dtype = arr.dtype.name
        else:
            val_min = float(np.min(arr))
            val_max = float(np.max(arr))
            dtype = arr.dtype.name
        return str(arr.shape), dtype, f"[{val_min:.2f}, {val_max:.2f}]"

    if HAS_RICH:
        table = Table(title="Data Summary", box=None)
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Shape", style="magenta")
        table.add_column("Dtype", style="green")
        table.add_column("Range", style="yellow")

        for name, arr in [("Input Volume 1", v1), ("Input Volume 2", v2), ("Interpolated", interp)]:
             shape, dtype, val_range = get_info(arr)
             table.add_row(name, shape, dtype, val_range)

        console.print(table)
        console.print()
    else:
        print("\n📊 Data Summary:")
        for name, arr in [("Input Volume 1", v1), ("Input Volume 2", v2), ("Interpolated", interp)]:
             shape, dtype, val_range = get_info(arr)
             print(f"   • {name:<15} Shape: {shape:<20} Dtype: {dtype:<8} Range: {val_range}")
        print()

def create_dummy_3d_data(shape: tuple = (1, 10, 64, 64, 1), num_sticks: int = 5, stick_length: int = 5, seed: int = 1234) -> np.ndarray:
    """Creates dummy 3D volumetric data containing simple 'sticks'."""
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
    print_header("3D Volume Interpolation Example")

    gpus = tf.config.list_physical_devices('GPU')
    if HAS_RICH:
        if gpus: console.print(f"[bold green]🚀 Running on GPU:[/bold green] {gpus[0].name}")
        else: console.print("[bold yellow]🐢 Running on CPU[/bold yellow] [dim](Performance might be slower)[/dim]")
    else:
        print(f"🚀 Running on GPU: {gpus[0].name}" if gpus else "🐢 Running on CPU (Performance might be slower)")

    try:
        with print_status("[1/4] Loading FILM model..."):
            interpolator_3d = Interpolator3D()
        print_success("FILM model loaded successfully!")
    except Exception as e:
        print_error(f"Error loading FILM model: {e}")
        exit(1)

    with print_status("[2/4] Creating dummy 3D data..."):
        volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), seed=1234)
        volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), seed=5678)
    print_success("Dummy data created.")

    dt = np.array([0.5], dtype=np.float32)
    start_time = time.time()
    with print_status("[3/4] Interpolating 3D volumes...", spinner="runner"):
        interpolated_volume = interpolator_3d(volume1, volume2, dt)
    elapsed = time.time() - start_time

    if HAS_RICH:
        console.print(f"[bold green]✅ Interpolation complete![/bold green] [dim]({elapsed:.2f}s)[/dim]")
    else:
        print(f"✅ Interpolation complete! ({elapsed:.2f}s)")

    print_summary(volume1, volume2, interpolated_volume)

    with print_status("[4/4] Performing Maximum Intensity Projection..."):
        mip_v1 = max_intensity_projection(volume1, axis=1)
        mip_interp = max_intensity_projection(interpolated_volume, axis=1)
        mip_v2 = max_intensity_projection(volume2, axis=1)

        out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'interpolated_mip_comparison.png')

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(mip_v1[0, :, :, 0], 0, 1), cmap='gray', vmin=0, vmax=1)
        plt.title("Start (t=0)")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(mip_interp[0], 0, 1))
        plt.title("Interpolated (t=0.5)")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(np.clip(mip_v2[0, :, :, 0], 0, 1), cmap='gray', vmin=0, vmax=1)
        plt.title("End (t=1)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path)

    if HAS_RICH:
        console.print(f"[bold green]✨ Saved comparison MIP image to:[/bold green] [link=file://{out_path}]{out_path}[/link]")
    else:
        print(f"✨ Saved comparison MIP image to: {out_path}")
