from unittest import TestCase

from ecopt.hyperparameter import Hyperparameter, Range, Choice, Fixed


class TestHyperparameter(TestCase):
    """Unit tests for the ecopt.hyperparameter.Hyperparameter class."""

    def test_cannot_init(self):
        """Test that initailising raises an exception."""
        self.assertRaises(NotImplementedError, Hyperparameter)


class TestRange(TestCase):
    """Unit tests for the ecopt.hyperparameter.Range class."""

    def test_init(self):
        """Test that initailising works as expected."""
        hyperparameter = Range(value=0.5, min=0, max=1)
        self.assertEqual(0.5, hyperparameter.value)
