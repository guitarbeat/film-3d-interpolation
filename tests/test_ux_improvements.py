import unittest
from unittest.mock import patch
from film_3d import Interpolator3D

class TestUXImprovements(unittest.TestCase):
    """Tests for UX improvements in the film_3d module."""

    @patch('film_3d.hub.load')
    def test_interpolator_3d_init_error_handling(self, mock_load):
        """Test that Interpolator3D raises a helpful RuntimeError on network failure."""
        # Simulate a network error (OSError)
        mock_load.side_effect = OSError("Network unreachable")

        with self.assertRaises(RuntimeError) as cm:
            Interpolator3D()

        # Check that the error message is helpful
        self.assertIn("Failed to load FILM model", str(cm.exception))
        self.assertIn("check your internet connection", str(cm.exception))

        # Verify the original exception is chained (optional but good practice)
        # Note: In Python 3, __cause__ or __context__ holds the chained exception.
        # self.assertIsInstance(cm.exception.__cause__, OSError)

    @patch('film_3d.hub.load')
    def test_interpolator_3d_init_value_error_handling(self, mock_load):
        """Test that Interpolator3D raises a helpful RuntimeError on ValueError (e.g. bad handle)."""
        # Simulate a ValueError
        mock_load.side_effect = ValueError("Invalid handle")

        with self.assertRaises(RuntimeError) as cm:
            Interpolator3D()

        self.assertIn("Failed to load FILM model", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
