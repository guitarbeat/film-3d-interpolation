import unittest
import numpy as np
from film_3d import Interpolator3D, max_intensity_projection, _pad_to_align

class TestUXImprovements(unittest.TestCase):
    """Tests focusing on UX improvements, specifically error handling."""

    def test_interpolator_3d_input_validation(self):
        """Test that Interpolator3D raises ValueError for invalid input shapes."""
        interpolator = Interpolator3D()

        # 4D input (invalid for __call__)
        x0 = np.zeros((1, 64, 64, 3), dtype=np.float32)
        x1 = np.zeros((1, 64, 64, 3), dtype=np.float32)
        dt = np.array([0.5], dtype=np.float32)

        with self.assertRaises(ValueError) as cm:
            interpolator(x0, x1, dt)
        self.assertIn("Input volumes must be 5D", str(cm.exception))

        # Mismatched depth
        x0 = np.zeros((1, 10, 64, 64, 3), dtype=np.float32)
        x1 = np.zeros((1, 12, 64, 64, 3), dtype=np.float32) # Different depth

        with self.assertRaises(ValueError) as cm:
            interpolator(x0, x1, dt)
        self.assertIn("same depth dimension", str(cm.exception))

    def test_mip_input_validation(self):
        """Test that max_intensity_projection raises ValueError for invalid input."""
        # 4D input
        volume = np.zeros((1, 64, 64, 3), dtype=np.float32)

        with self.assertRaises(ValueError) as cm:
            max_intensity_projection(volume)
        self.assertIn("Input volume must be 5D", str(cm.exception))

    def test_pad_to_align_validation(self):
        """Test _pad_to_align internal validation."""
        # 3D input (invalid, expects 4D or 5D)
        x = np.zeros((64, 64, 3), dtype=np.float32)
        align = 16

        with self.assertRaises(ValueError) as cm:
            _pad_to_align(x, align)
        self.assertIn("Input must be 4D", str(cm.exception))

        # Invalid align
        x_valid = np.zeros((1, 64, 64, 3), dtype=np.float32)
        with self.assertRaises(ValueError) as cm:
            _pad_to_align(x_valid, 0)
        self.assertIn("Alignment value must be a positive number", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
