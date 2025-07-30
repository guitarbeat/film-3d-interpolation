
import numpy as np
from film_3d import Interpolator3D, max_intensity_projection
import matplotlib.pyplot as plt

def create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=10):
    """Creates dummy 3D data with 'sticks' for demonstration.
    Shape: (batch_size, depth, height, width, channels)
    """
    data = np.zeros(shape, dtype=np.float32)
    batch_size, depth, height, width, channels = shape

    for b in range(batch_size):
        for _ in range(num_sticks):
            # Random starting point for the stick
            start_d = np.random.randint(0, depth - stick_length)
            start_h = np.random.randint(0, height)
            start_w = np.random.randint(0, width)

            # Random orientation (for simplicity, mostly vertical or horizontal within a slice)
            orientation = np.random.choice([0, 1, 2]) # 0: depth, 1: height, 2: width

            for i in range(stick_length):
                if orientation == 0: # Along depth
                    if start_d + i < depth:
                        data[b, start_d + i, start_h, start_w, 0] = 1.0
                elif orientation == 1: # Along height
                    if start_h + i < height:
                        data[b, start_d, start_h + i, start_w, 0] = 1.0
                else: # Along width
                    if start_w + i < width:
                        data[b, start_d, start_h, start_w + i, 0] = 1.0
    return data

if __name__ == '__main__':
    interpolator_3d = Interpolator3D()

    # Create two dummy 3D volumes representing two time points
    print("Creating dummy 3D data...")
    volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5)
    volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5)

    # Interpolate at midway
    dt = np.array([0.5], dtype=np.float32)

    print("Interpolating 3D volumes...")
    interpolated_volume = interpolator_3d(volume1, volume2, dt)
    print("Interpolation complete. Interpolated volume shape:", interpolated_volume.shape)

    # Perform Maximum Intensity Projection on the interpolated volume along the depth axis
    print("Performing Maximum Intensity Projection...")
    mip_image = max_intensity_projection(interpolated_volume, axis=1) # MIP along depth axis
    print("MIP image shape:", mip_image.shape)

    # Visualize the MIP image
    plt.figure(figsize=(6, 6))
    plt.imshow(mip_image[0, :, :, 0], cmap='gray') # Assuming batch size 1 and 1 channel
    plt.title("Maximum Intensity Projection of Interpolated Volume")
    plt.colorbar()
    plt.savefig("interpolated_mip.png")
    plt.show()

    print("MIP example script finished. See interpolated_mip.png")


