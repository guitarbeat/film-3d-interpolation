import time
import os
import sys
import numpy as np
import tensorflow as tf
import warnings

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Add src to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from film_3d import Interpolator3D

def create_dummy_3d_data(shape: tuple, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float32)

def benchmark():
    print("Setting up benchmark...")
    interpolator = Interpolator3D(align=64)

    # Create input data (similar to what's used in the example, but maybe larger to see the diff better)
    # The example used (1, 10, 64, 64, 1). Let's use something slightly substantial but not too slow.
    # 128x128 is mentioned in memory as a good size.
    shape = (1, 16, 128, 128, 1)

    volume1 = create_dummy_3d_data(shape, seed=1234)
    volume2 = create_dummy_3d_data(shape, seed=5678)
    dt = np.array([0.5], dtype=np.float32)

    # Warmup
    print("Warming up...")
    interpolator(volume1, volume2, dt)

    print("Running benchmark...")
    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        interpolator(volume1, volume2, dt)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"Average time per interpolation: {avg_time:.4f} seconds")

if __name__ == '__main__':
    benchmark()
