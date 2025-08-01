
import numpy as np
import matplotlib.pyplot as plt
from film_3d import Interpolator3D, max_intensity_projection

def create_dummy_3d_data(shape: tuple = (1, 10, 64, 64, 1), num_sticks: int = 5, stick_length: int = 5) -> np.ndarray:
    """Creates dummy 3D volumetric data containing simple 'sticks' for demonstration purposes.

    This function generates a 5D NumPy array representing a batch of 3D volumes.
    Each volume contains a specified number of randomly placed 'sticks' (lines of high intensity).
    This is useful for testing 3D interpolation and MIP functions without needing real data.

    Args:
        shape: A tuple specifying the shape of the output data:
               (batch_size, depth, height, width, channels).
        num_sticks: The number of 'sticks' to generate within each 3D volume.
        stick_length: The length of each 'stick' in voxels.

    Returns:
        A NumPy array of the specified shape, filled with zeros except for the
        'sticks', which have a value of 1.0. Data type is np.float32.
    """
    data = np.zeros(shape, dtype=np.float32)
    batch_size, depth, height, width, channels = shape

    for b in range(batch_size):
        for _ in range(num_sticks):
            # Randomly determine the starting coordinates for a stick.
            # Ensure there's enough space for the stick to fit within the volume.
            start_d = np.random.randint(0, depth - stick_length + 1)
            start_h = np.random.randint(0, height)
            start_w = np.random.randint(0, width)

            # Randomly choose an orientation for the stick (along depth, height, or width).
            orientation = np.random.choice([0, 1, 2]) # 0: depth, 1: height, 2: width

            # Place the stick voxels into the dummy data array.
            for i in range(stick_length):
                if orientation == 0: # Stick extends along the depth (Z) axis.
                    if start_d + i < depth:
                        data[b, start_d + i, start_h, start_w, 0] = 1.0
                elif orientation == 1: # Stick extends along the height (Y) axis.
                    if start_h + i < height:
                        data[b, start_d, start_h + i, start_w, 0] = 1.0
                else: # Stick extends along the width (X) axis.
                    if start_w + i < width:
                        data[b, start_d, start_h, start_w + i, 0] = 1.0
    return data

if __name__ == '__main__':
    # Initialize the 3D interpolator.
    interpolator_3d = Interpolator3D()

    # Create two dummy 3D volumes representing two different time points.
    # These volumes will be used as input for the interpolation.
    print("Creating dummy 3D data...")
    volume1 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5)
    volume2 = create_dummy_3d_data(shape=(1, 10, 64, 64, 1), num_sticks=5, stick_length=5)

    # Define the interpolation time. A value of 0.5 means interpolating exactly
    # midway between volume1 and volume2.
    dt = np.array([0.5], dtype=np.float32)

    print("Interpolating 3D volumes...")
    # Perform the 3D frame interpolation.
    interpolated_volume = interpolator_3d(volume1, volume2, dt)
    print("Interpolation complete. Interpolated volume shape:", interpolated_volume.shape)

    # Perform Maximum Intensity Projection (MIP) on the interpolated 3D volume.
    # MIP is performed along the depth (axis=1) to get a 2D representation.
    print("Performing Maximum Intensity Projection...")
    mip_image = max_intensity_projection(interpolated_volume, axis=1) # MIP along depth axis
    print("MIP image shape:", mip_image.shape)

    # Visualize and save the resulting MIP image.
    plt.figure(figsize=(6, 6))
    # Display the 2D MIP image. Assuming batch size 1 and 1 channel for simplicity.
    plt.imshow(mip_image[0, :, :, 0], cmap='gray')
    plt.title("Maximum Intensity Projection of Interpolated Volume")
    plt.colorbar()
    plt.savefig("interpolated_mip.png") # Save the image to a file.
    plt.show() # Display the image (might not be visible in all environments).

    print("MIP example script finished. See interpolated_mip.png for the output.")


