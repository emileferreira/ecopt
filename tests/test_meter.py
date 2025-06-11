from unittest import TestCase

from ecopt.meter import Meter

from .test_model import DummyModel


class TestMeter(TestCase):
    """Unit tests for the ecopt.meter.Meter class."""

    def test_cannot_call(self):
        """Test that calling raises an exception."""
        model = DummyModel()
        meter = Meter()
        with self.assertRaises(NotImplementedError):
            meter(model)
