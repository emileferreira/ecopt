from unittest import TestCase

from ecopt.meter import Meter, CodeCarbonMeter

from .test_model import DummyModel


class TestMeter(TestCase):
    """Unit tests for the ecopt.meter.Meter class."""

    def test_cannot_call(self):
        """Test that calling raises an exception."""
        model = DummyModel()
        meter = Meter()
        with self.assertRaises(NotImplementedError):
            meter(model)


class TestCodeCarbonMeter(TestCase):
    """Unit tests for the ecopt.meter.CodeCarbonMeter class."""

    def test_metrics(self):
        """Test that the meter returns the expected metrics."""
        model = DummyModel()
        utility_measure = "f1"
        cost_measures = ["train_energy", "train_carbon", "train_time",
                         "evaluate_energy", "evaluate_carbon", "evaluate_time",
                         "samples_per_wh", "samples_per_j", "samples_per_kg",
                         "samples_per_s"]
        meter = CodeCarbonMeter(log_level="ERROR")
        observation = meter(model, utility_measure=utility_measure)
        self.assertTrue(utility_measure in observation.keys())
        for measure in cost_measures:
            self.assertTrue(measure in observation.keys())
        self.assertEqual(len(cost_measures) + 1, len(observation.keys()))
