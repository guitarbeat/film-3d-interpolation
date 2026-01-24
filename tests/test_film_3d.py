import unittest
import numpy as np
from film_3d import Interpolator3D, max_intensity_projection, load_volume, _pad_to_align


class TestFilm3D(unittest.TestCase):
    """Unit tests for the film_3d module.

    This class contains tests for the `load_volume` and `max_intensity_projection`
    functions. Testing `Interpolator3D` directly is more complex as it relies on
    a TensorFlow Hub model, which would require mocking for isolated unit tests.
    """

    def test_load_volume(self):
        """Tests the `load_volume` function.

        Verifies that the function returns a NumPy array with the expected
        shape and data type, based on its current dummy implementation.
        """
        volume = load_volume("dummy_path.tif")
        self.assertIsInstance(volume, np.ndarray)
        # The dummy implementation returns a (10, 128, 128, 1) shaped array.
        self.assertEqual(volume.shape, (10, 128, 128, 1))
        self.assertEqual(volume.dtype, np.float32)

    def test_max_intensity_projection(self):
        """Tests the `max_intensity_projection` function.

        Verifies that the function correctly performs Maximum Intensity Projection
        along the specified axis for both simple and more complex 3D arrays.
        """
        # Test case 1: Simple 3D array
        # Input shape: (batch, depth, height, width, channels)
        test_volume = np.array([
            [[[[1], [2]], [[3], [4]]]],
            [[[[5], [6]], [[7], [8]]]]
        ], dtype=np.float32).reshape(1, 2, 2, 2, 1)

        # Perform MIP along the depth axis (axis=1).
        mip_result = max_intensity_projection(test_volume, axis=1)
        # Expected result after MIP: max of (1,5), (2,6), (3,7), (4,8)
        expected_mip = np.array([
            [[[5], [6]], [[7], [8]]]
        ], dtype=np.float32).reshape(1, 2, 2, 1)
        np.testing.assert_array_equal(mip_result, expected_mip)

        # Test case 2: More complex 3D array with varying max values
        complex_volume = np.zeros((1, 3, 5, 5, 1), dtype=np.float32)
        # Set specific voxels to be the maximum in their respective (x,y) positions.
        complex_volume[0, 0, 2, 2, 0] = 10  # Max in first slice at (2,2)
        complex_volume[0, 1, 1, 1, 0] = 20  # Max in second slice at (1,1)
        complex_volume[0, 2, 3, 3, 0] = 5   # Max in third slice at (3,3)

        mip_complex = max_intensity_projection(complex_volume, axis=1)
        expected_complex_mip = np.zeros((1, 5, 5, 1), dtype=np.float32)
        expected_complex_mip[0, 2, 2, 0] = 10
        expected_complex_mip[0, 1, 1, 0] = 20
        expected_complex_mip[0, 3, 3, 0] = 5

        np.testing.assert_array_equal(mip_complex, expected_complex_mip)

    def test_interpolator_3d(self):
        """Tests the Interpolator3D class.

        Verifies that the interpolator can process a small 3D volume
        and return the correct shape.
        """
        interpolator = Interpolator3D(align=16)
        batch_size = 1
        depth = 4
        height = 64
        width = 64
        channels = 1

        x0 = np.random.rand(batch_size, depth, height, width, channels).astype(np.float32)
        x1 = np.random.rand(batch_size, depth, height, width, channels).astype(np.float32)
        dt = np.array([0.5], dtype=np.float32)

        result = interpolator(x0, x1, dt)

        # Expected output channels is 3 because FILM model outputs RGB
        expected_shape = (batch_size, depth, height, width, 3)
        self.assertEqual(result.shape, expected_shape)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)

    def test_interpolator_3d_return_tensor(self):
        """Tests the Interpolator3D class with return_tensor=True."""
        import tensorflow as tf
        interpolator = Interpolator3D(align=16)
        batch_size = 1
        depth = 4
        height = 64
        width = 64
        channels = 1

        x0 = np.random.rand(batch_size, depth, height, width, channels).astype(np.float32)
        x1 = np.random.rand(batch_size, depth, height, width, channels).astype(np.float32)
        dt = np.array([0.5], dtype=np.float32)

        result_tensor = interpolator(x0, x1, dt, return_tensor=True)

        self.assertTrue(tf.is_tensor(result_tensor))
        # Expected output channels is 3 because FILM model outputs RGB
        expected_shape = (batch_size, depth, height, width, 3)
        self.assertEqual(result_tensor.shape, expected_shape)

        # Verify values match numpy return (within tolerance)
        result_numpy = interpolator(x0, x1, dt, return_tensor=False)
        np.testing.assert_allclose(result_tensor.numpy(), result_numpy, atol=1e-5)

    def test_max_intensity_projection_tensor(self):
        """Tests max_intensity_projection with Tensor input."""
        import tensorflow as tf

        # Test case 1: Simple 3D array as Tensor
        test_volume_np = np.array([
            [[[[1], [2]], [[3], [4]]]],
            [[[[5], [6]], [[7], [8]]]]
        ], dtype=np.float32).reshape(1, 2, 2, 2, 1)
        test_volume_tensor = tf.convert_to_tensor(test_volume_np)

        # Perform MIP
        mip_result = max_intensity_projection(test_volume_tensor, axis=1)

        self.assertTrue(tf.is_tensor(mip_result))

        expected_mip = np.array([
            [[[5], [6]], [[7], [8]]]
        ], dtype=np.float32).reshape(1, 2, 2, 1)

        np.testing.assert_array_equal(mip_result.numpy(), expected_mip)

    def test_pad_to_align(self):
        """Tests the _pad_to_align function for correctness."""
        import tensorflow as tf

        align = 4
        # Case 1: 4D input, needs padding
        # Shape: (1, 2, 2, 1) -> Pad to (1, 4, 4, 1)
        x_4d = np.ones((1, 2, 2, 1), dtype=np.float32)
        padded_4d, bbox = _pad_to_align(x_4d, align)

        self.assertEqual(padded_4d.shape, (1, 4, 4, 1))
        # Verify padding logic: 2 -> 4 requires 2 padding.
        # offset should be 2//2 = 1.
        self.assertEqual(bbox['offset_height'], 1)
        self.assertEqual(bbox['offset_width'], 1)
        self.assertEqual(bbox['target_height'], 2)
        self.assertEqual(bbox['target_width'], 2)

        # Case 2: 4D input, already aligned
        x_4d_aligned = np.ones((1, 4, 4, 1), dtype=np.float32)
        padded_4d_aligned, bbox_aligned = _pad_to_align(x_4d_aligned, align)
        self.assertEqual(padded_4d_aligned.shape, (1, 4, 4, 1))
        self.assertIsNone(bbox_aligned)

        # Case 3: 5D input, needs padding
        # Shape: (1, 2, 2, 2, 1) -> Pad to (1, 2, 4, 4, 1)
        x_5d = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
        padded_5d, bbox_5d = _pad_to_align(x_5d, align)

        self.assertEqual(padded_5d.shape, (1, 2, 4, 4, 1))
        self.assertEqual(bbox_5d['offset_height'], 1)
        self.assertEqual(bbox_5d['offset_width'], 1)

        # Verify content logic (padding should be zeros)
        # Center crop should be ones
        center_crop = tf.image.crop_to_bounding_box(padded_5d[:, 0, :, :, :], **bbox_5d)
        np.testing.assert_array_equal(center_crop.numpy(), np.ones((1, 2, 2, 1)))

        # Case 4: 5D input, already aligned
        x_5d_aligned = np.ones((1, 2, 4, 4, 1), dtype=np.float32)
        padded_5d_aligned, bbox_5d_aligned = _pad_to_align(x_5d_aligned, align)
        self.assertEqual(padded_5d_aligned.shape, (1, 2, 4, 4, 1))
        self.assertIsNone(bbox_5d_aligned)


if __name__ == '__main__':
    unittest.main()