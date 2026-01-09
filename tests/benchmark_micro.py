
import time
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def micro_benchmark():
    print("Running micro-benchmark: tf.repeat vs tf.image.grayscale_to_rgb")

    # Shape: (Batch*Depth, H, W, 1)
    # Using a larger batch to see the difference clearly
    shape = (100, 512, 512, 1)
    x = tf.random.uniform(shape)

    # Warmup
    for _ in range(5):
        _ = tf.repeat(x, 3, axis=-1)
        _ = tf.image.grayscale_to_rgb(x)

    iterations = 100

    # Test tf.repeat
    start = time.time()
    for _ in range(iterations):
        _ = tf.repeat(x, 3, axis=-1)
    end = time.time()
    repeat_time = (end - start) / iterations
    print(f"tf.repeat average time: {repeat_time:.6f} s")

    # Test tf.image.grayscale_to_rgb
    start = time.time()
    for _ in range(iterations):
        _ = tf.image.grayscale_to_rgb(x)
    end = time.time()
    rgb_time = (end - start) / iterations
    print(f"tf.image.grayscale_to_rgb average time: {rgb_time:.6f} s")

    improvement = (repeat_time - rgb_time) / repeat_time * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    micro_benchmark()
