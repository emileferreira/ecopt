from enum import Enum


class ValueType(str, Enum):
    """The supported hyperparameter value types."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"


class Hyperparameter:
    """An abstract hyperparameter for optimisation."""

    def __init__(self):
        """Construct an abstract hyperparameter."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a str representation of the hyperparameter."""
        return str(self.value)


class Range(Hyperparameter):
    """A range hyperparameter for optimisation."""

    def __init__(self, value: float | int,
                 min: float | int,
                 max: float | int,
                 log_scale: bool = False,
                 value_type: ValueType = None):
        """Construct a new range hyperparameter."""
        if value < min or value > max:
            raise ValueError('Provided value outside of range.')
        supported_types = (ValueType.FLOAT, ValueType.INT, None)
        if value_type not in supported_types:
            raise ValueError('Range only supports int and float value types.')
        self.value = value
        self.min = min
        self.max = max
        self.log_scale = log_scale
        self.value_type = value_type

    def to_dict(self, name: str) -> dict:
        """Return a dict representation for use with Ax."""
        return {
            "name": name,
            "type": "range",
            "bounds": [self.min, self.max],
            "log_scale": self.log_scale,
            "value_type": self.value_type.value
            if self.value_type is not None else None
        }


class Choice(Hyperparameter):
    """A choice hyperparameter for optimisation."""

    def __init__(self, value: float | int | bool | str,
                 values: list[float | int | bool | str],
                 is_ordered: bool = True, sort_values: bool = False,
                 value_type: ValueType = None):
        """Construct a new choice hyperparameter."""
        if value not in values:
            raise ValueError('Provided value not in choices.')
        self.value = value
        self.values = values
        self.is_ordered = is_ordered
        self.sort_values = sort_values
        self.value_type = value_type

    def to_dict(self, name: str) -> dict:
        """Return a dict representation for use with Ax."""
        return {
            "name": name,
            "type": "choice",
            "values": self.values,
            "is_ordered": self.is_ordered,
            "sort_values": self.sort_values,
            "value_type": self.value_type.value
            if self.value_type is not None else None
        }


class Fixed(Hyperparameter):
    """A fixed hyperparameter."""

    def __init__(self, value: float | int | bool | str):
        """Construct a new fixed hyperparameter."""
        self.value = value

    def to_dict(self, name: str) -> dict:
        """Return a dict representation for use with Ax."""
        return {"name": name, "type": "fixed", "value": self.value}
