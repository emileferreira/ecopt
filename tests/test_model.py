from unittest import TestCase

from ecopt.model import Model
from ecopt.hyperparameter import Fixed, Choice, Hyperparameter


class DummyModel(Model):
    """A dummy model to test the meter."""

    def __init__(self):
        """Construct the model using the (potentially updated)
        hyperparameters."""
        self.batch_size = Choice(64, values=[32, 64, 128])
        self.num_epochs = Fixed(10)

    def train(self):
        """Train the model."""
        pass

    def evaluate(self) -> (float, int):
        """
        Evaluate the model.

        :return: The measured performance and the number of samples
        """
        return 1, 1


class TestModel(TestCase):
    """Unit tests for the ecopt.model.Model class."""

    def test_cannot_init(self):
        """Test that init raises an exception."""
        with self.assertRaises(NotImplementedError):
            Model()


class TestDummyModel(TestCase):
    """Unit tests for the DummyModel class."""

    def test_hyperparameters(self):
        """Test that the hyperparameters are returned."""
        model = DummyModel()
        hyperparameters = model.hyperparameters
        self.assertEqual(2, len(hyperparameters))
        self.assertTrue("batch_size" in hyperparameters.keys())
        self.assertTrue("num_epochs" in hyperparameters.keys())
        for value in hyperparameters.values():
            self.assertIsInstance(value, Hyperparameter)
