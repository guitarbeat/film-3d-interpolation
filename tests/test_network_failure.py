import unittest
from unittest.mock import patch
from film_3d import Interpolator3D

class TestNetworkFailure(unittest.TestCase):
    @patch('film_3d.hub.load')
    def test_init_network_failure(self, mock_load):
        # Simulate network error (OSError)
        mock_load.side_effect = OSError("Network is unreachable")

        with self.assertRaises(RuntimeError) as cm:
            Interpolator3D()

        # Verify the error message contains actionable advice
        msg = str(cm.exception)
        self.assertIn("Failed to load FILM model", msg)
        self.assertIn("internet connection", msg)

    @patch('film_3d.hub.load')
    def test_init_value_error(self, mock_load):
        # Simulate loading error (ValueError - e.g. bad handle)
        mock_load.side_effect = ValueError("Invalid handle")

        with self.assertRaises(RuntimeError) as cm:
            Interpolator3D()

        msg = str(cm.exception)
        self.assertIn("Failed to load FILM model", msg)
