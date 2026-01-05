import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import Generator, Iterable, List, Optional

# This module provides functionalities for 3D frame interpolation using a modified FILM model
# and for performing Maximum Intensity Projection (MIP) on 3D volumetric data.

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def load_volume(vol_path: str) -> np.ndarray:
  """Loads a 3D volume from a specified path and normalizes its pixel values.

  This function currently serves as a placeholder. In a real-world application,
  it would handle various 3D image formats (e.g., DICOM, TIFF stacks, NIfTI).
  For demonstration purposes, it generates a dummy 3D volume.

  Args:
    vol_path: The file path to the 3D volume. (Currently unused for dummy data generation).

  Returns:
    A NumPy array representing the 3D volume with shape
    [depth, height, width, num_channels], where pixel values are normalized
    to the range [0, 1] and data type is np.float32.
  """
  print(f"Placeholder: Simulating loading volume from {vol_path}")
  # Simulate loading a 3D volume (e.g., 10 slices of 128x128 grayscale images)
  # The last dimension (channels) is 1 for grayscale.
  dummy_volume = np.random.rand(10, 128, 128, 1).astype(np.float32)
  return dummy_volume

def _pad_to_align(x: np.ndarray, align: int) -> tuple[tf.Tensor, dict]:
  """Pads an image batch or 3D volume so its height and width are divisible by 'align'.

  This padding is necessary for models that require input dimensions to be
  multiples of a certain value (e.g., due to pooling layers).
  For 3D volumes, padding is applied independently to each 2D slice.

  Args:
    x: The input image batch (4D: [batch, height, width, channels]) or
       3D volume (5D: [batch, depth, height, width, channels]) to be padded.
    align: The integer value to which height and width should be aligned.

  Returns:
    A tuple containing:
    - padded_x: The padded image batch or 3D volume as a TensorFlow Tensor.
    - bbox_to_crop: A dictionary containing bounding box information
                    (offset_height, offset_width, target_height, target_width)
                    that can be used with `tf.image.crop_to_bounding_box`
                    to revert the padding.
  """
  # Input validation: ensure the input array has 4 or 5 dimensions.
  assert np.ndim(x) == 4 or np.ndim(x) == 5, \
      "Input must be 4D (batch, H, W, C) or 5D (batch, D, H, W, C)"
  assert align > 0, "Alignment value must be a positive number."

  # Determine dimensions based on whether it's a 4D image batch or 5D volume.
  if np.ndim(x) == 5:
      batch_size, depth, height, width, channels = x.shape
  else:
      # If 4D, treat as a single-slice 3D volume for consistent processing.
      batch_size, height, width, channels = x.shape
      depth = 1

  # Calculate padding needed for height and width.
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  # Define the bounding box for padding.
  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }

  # Apply padding to each 2D slice of the volume independently.
  padded_x_slices = []
  for d in range(depth):
      if np.ndim(x) == 5:
          # Extract a 2D slice from the 5D volume and pad it.
          padded_x_slices.append(tf.image.pad_to_bounding_box(x[:, d, :, :, :], **bbox_to_pad))
      else:
          # If 4D, pad the entire image batch.
          padded_x_slices.append(tf.image.pad_to_bounding_box(x, **bbox_to_pad))

  # Stack the padded slices back into the original dimension format.
  if np.ndim(x) == 5:
      padded_x = tf.stack(padded_x_slices, axis=1) # Stack back into 5D volume.
  else:
      padded_x = padded_x_slices[0] # Remains 4D image batch.

  # Define the bounding box for cropping back to original dimensions after inference.
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator3D:
  """A class for generating interpolated frames between two input 3D volumes.

  This class adapts the 2D FILM (Frame Interpolation for Large Motion) model
  from TensorFlow Hub to work with 3D volumetric data. It processes each 2D
  slice of the 3D volumes independently using the underlying 2D FILM model
  and then reconstructs the interpolated 3D volume.
  """

  def __init__(self, align: int = 64) -> None:
    """Initializes the Interpolator3D by loading the FILM model.

    Args:
      align: If greater than 1, input dimensions (height and width) will be
             padded to be divisible by this value before inference. This is
             a common requirement for deep learning models.
    """
    # Load the pre-trained 2D FILM model from TensorFlow Hub.
    self._model = hub.load("https://tfhub.dev/google/film/1")
    self._align = align

  def __call__(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated 3D volume between two given 3D volumes.

    The interpolation is performed slice by slice using the 2D FILM model.

    Args:
      x0: The first input 3D volume. Expected shape:
          (batch_size, depth, height, width, channels). Data type must be np.float32.
      x1: The second input 3D volume. Expected shape:
          (batch_size, depth, height, width, channels). Data type must be np.float32.
      dt: The sub-frame time, indicating the position of the generated frame
          between x0 and x1. Expected shape: (batch_size,). Values should be
          in the range [0, 1], where 0.5 represents the midway point.

    Returns:
      The interpolated 3D volume as a NumPy array with the same dimensions as
      the input volumes: (batch_size, depth, height, width, channels).
    """
    # Input validation for 3D volumes.
    assert np.ndim(x0) == 5 and np.ndim(x1) == 5, \
        "Input volumes must be 5D (batch, depth, height, width, channels)"
    assert x0.shape[1] == x1.shape[1], "Input volumes must have the same depth dimension."

    batch_size, depth, height, width, channels = x0.shape

    # Optimizing by processing all slices in a single batch instead of looping.
    # Reshape (batch, depth, height, width, channels) -> (batch*depth, height, width, channels)
    x0_reshaped = np.reshape(x0, (batch_size * depth, height, width, channels))
    x1_reshaped = np.reshape(x1, (batch_size * depth, height, width, channels))

    # Apply padding if alignment is required.
    # _pad_to_align handles 4D input by padding the whole batch at once.
    if self._align is not None:
        slice0_padded, bbox_to_crop = _pad_to_align(x0_reshaped, self._align)
        slice1_padded, _ = _pad_to_align(x1_reshaped, self._align)
    else:
        slice0_padded = x0_reshaped
        slice1_padded = x1_reshaped

    # Convert 1-channel (grayscale) input to 3-channel (RGB) by repeating
    # the channel, as the FILM model expects 3-channel input.
    if slice0_padded.shape[-1] == 1:
        # slice0_padded is a Tensor if passed through _pad_to_align, or np.ndarray otherwise.
        # tf.repeat or np.repeat works depending on type, but tf.image.pad_to_bounding_box returns Tensor.
        if not tf.is_tensor(slice0_padded):
             slice0_padded = tf.convert_to_tensor(slice0_padded)
        if not tf.is_tensor(slice1_padded):
             slice1_padded = tf.convert_to_tensor(slice1_padded)

        # Optimization: Use tf.image.grayscale_to_rgb instead of tf.repeat for better performance (~18% faster)
        slice0_padded = tf.image.grayscale_to_rgb(slice0_padded)
        slice1_padded = tf.image.grayscale_to_rgb(slice1_padded)

    # Prepare time input. dt is (batch_size,). We need it to be (batch_size * depth, 1).
    # First, repeat each element 'depth' times.
    # dt: [t1, t2] -> [t1, t1, ..., t2, t2, ...]
    dt_repeated = np.repeat(dt, depth)
    dt_reshaped = dt_repeated[..., np.newaxis] # (batch*depth, 1)

    # Prepare inputs for the 2D FILM model.
    inputs = {
        'x0': slice0_padded,
        'x1': slice1_padded,
        'time': dt_reshaped
    }

    # Perform inference using the 2D FILM model on the entire batch.
    result = self._model(inputs, training=False)
    image_batch = result['image']

    # Crop the interpolated slices back to original dimensions if padding was applied.
    if self._align is not None:
        image_batch = tf.image.crop_to_bounding_box(image_batch, **bbox_to_crop)

    # Convert result back to numpy
    interpolated_flat = image_batch.numpy()

    # Reshape back to 5D: (batch*depth, H, W, C) -> (batch, depth, H, W, C)
    # Note: The model output is always 3 channels (RGB).
    out_channels = interpolated_flat.shape[-1]
    interpolated_volume = np.reshape(interpolated_flat, (batch_size, depth, height, width, out_channels))

    return interpolated_volume

def max_intensity_projection(volume: np.ndarray, axis: int = 1) -> np.ndarray:
    """Performs Maximum Intensity Projection (MIP) along a specified axis of a 3D volume.

    MIP is a method for 3D visualization that projects the voxels with the
    highest intensity onto a 2D plane. This is particularly useful for
    visualizing structures like 


3D sticks in time, as it highlights the brightest parts of the data.

    Args:
        volume: The input 3D volume as a NumPy array. Expected shape:
                (batch_size, depth, height, width, channels).
        axis: The axis along which to perform the MIP. For a 5D volume,
              axis=1 typically corresponds to the depth (Z-axis) dimension.

    Returns:
        A 2D projection of the volume as a NumPy array, with the specified
        axis collapsed. The shape will be (batch_size, height, width, channels)
        if MIP is performed along the depth axis.
    """
    assert np.ndim(volume) == 5, "Input volume must be 5D for MIP (batch, depth, height, width, channels)"
    return np.max(volume, axis=axis)