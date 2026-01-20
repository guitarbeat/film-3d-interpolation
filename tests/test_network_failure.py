import unittest
from unittest.mock import patch
import os
from film_3d import Interpolator3D

# Suppress TF logs for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TestNetworkFailure(unittest.TestCase):
    @patch('tensorflow_hub.load')
    def test_model_load_failure_message(self, mock_load):
        """Test that a helpful RuntimeError is raised when model download fails."""
        # Simulate a network error (OSError)
        mock_load.side_effect = OSError("Network is unreachable")

        with self.assertRaises(RuntimeError) as cm:
            Interpolator3D()

        # Verify the error message contains helpful actionable info
        error_msg = str(cm.exception)
        self.assertIn("Failed to load FILM model", error_msg)
        self.assertIn("internet connection", error_msg)
        self.assertIn("Original error: Network is unreachable", error_msg)

if __name__ == '__main__':
    unittest.main()
