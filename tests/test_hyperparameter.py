from unittest import TestCase

from ecopt.hyperparameter import (
    Hyperparameter,
    Range,
    Choice,
    Fixed,
    ValueType,
)


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

    def test_outside_range(self):
        """Test that a ValueError is raised for out-of-range values."""
        with self.assertRaises(ValueError):
            Range(value=1.5, min=0, max=1)
        with self.assertRaises(ValueError):
            Range(value=-0.5, min=0, max=1)

    def test_value_type(self):
        """Test that the correct str is returned as value type."""
        int_param = Range(value=0, min=0, max=1, value_type=ValueType.INT)
        self.assertEqual("int", int_param.to_dict("int_param")["value_type"])
        float_param = Range(value=0.5, min=0, max=1,
                            value_type=ValueType.FLOAT)
        self.assertEqual("float", float_param.to_dict(
            "float_param")["value_type"])
        with self.assertRaises(ValueError):
            Range(value=0, min=0, max=1, value_type=ValueType.STR)
        with self.assertRaises(ValueError):
            Range(value=0, min=0, max=1, value_type=ValueType.BOOL)


class TestChoice(TestCase):
    """Unit tests for the ecopt.hyperparameter.Choice class."""

    def test_init(self):
        """Test that initailising works as expected."""
        hyperparameter = Choice(value=True, values=[True, False])
        self.assertTrue(hyperparameter.value)

    def test_outside_choices(self):
        """Test that a ValueError is raised for invalid values."""
        with self.assertRaises(ValueError):
            Choice(value=3, values=[1, 2])

    def test_value_type(self):
        """Test that the correct str is returned as value type."""
        int_param = Choice(value=True, values=[True, False],
                           value_type=ValueType.INT)
        self.assertEqual("int", int_param.to_dict("int_param")["value_type"])
        float_param = Choice(value=True, values=[True, False],
                             value_type=ValueType.FLOAT)
        self.assertEqual("float", float_param.to_dict(
            "float_param")["value_type"])
        str_param = Choice(value=True, values=[True, False],
                           value_type=ValueType.STR)
        self.assertEqual("str", str_param.to_dict("str_param")["value_type"])
        bool_param = Choice(value=True, values=[True, False],
                            value_type=ValueType.BOOL)
        self.assertEqual("bool", bool_param.to_dict(
            "bool_param")["value_type"])


class TestFixed(TestCase):
    """Unit tests for the ecopt.hyperparameter.Choice class."""

    def test_init(self):
        """Test that initailising works as expected."""
        hyperparameter = Fixed(value=True)
        self.assertTrue(hyperparameter.value)
