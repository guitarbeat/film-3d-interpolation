import unittest
import numpy as np
import tensorflow as tf
from film_3d import Interpolator3D, max_intensity_projection, load_volume


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

    def test_max_intensity_projection_tensor(self):
        """Tests max_intensity_projection with TensorFlow tensor input."""
        test_volume = np.array([
            [[[[1], [2]], [[3], [4]]]],
            [[[[5], [6]], [[7], [8]]]]
        ], dtype=np.float32).reshape(1, 2, 2, 2, 1)
        test_tensor = tf.convert_to_tensor(test_volume)

        mip_result = max_intensity_projection(test_tensor, axis=1)

        self.assertTrue(tf.is_tensor(mip_result))

        expected_mip = np.array([
            [[[5], [6]], [[7], [8]]]
        ], dtype=np.float32).reshape(1, 2, 2, 1)

        np.testing.assert_array_equal(mip_result.numpy(), expected_mip)

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

        # Test default behavior (return numpy)
        result_np = interpolator(x0, x1, dt)
        expected_shape = (batch_size, depth, height, width, 3)
        self.assertEqual(result_np.shape, expected_shape)
        self.assertIsInstance(result_np, np.ndarray)
        self.assertEqual(result_np.dtype, np.float32)

        # Test return_tensor=True
        result_tf = interpolator(x0, x1, dt, return_tensor=True)
        self.assertEqual(result_tf.shape, expected_shape)
        self.assertTrue(tf.is_tensor(result_tf))
        self.assertEqual(result_tf.dtype, tf.float32)


if __name__ == '__main__':
    unittest.main()