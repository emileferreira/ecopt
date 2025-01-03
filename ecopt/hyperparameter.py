class Hyperparameter:
    """An abstract hyperparemeter for optimisation."""

    def __init__(self):
        """Construct an abstract hyperparemeter."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a str representation of the hyperparemeter."""
        return str(self.value)


class Range(Hyperparameter):
    """A range hyperparemeter for optimisation."""

    def __init__(self, value: float or int,
                 min: float or int,
                 max: float or int,
                 log_scale: bool = False):
        """Construct a new range hyperparemeter."""
        if value < min or value > max:
            raise ValueError('Provided value outside of range.')
        self.value, self.min, self.max, self.log_scale = (
            value, min, max, log_scale
        )

    def to_dict(self, name: str) -> dict:
        """Return a dict representation for use with Ax."""
        return {
            "name": name,
            "type": "range",
            "bounds": [self.min, self.max],
            "log_scale": self.log_scale
        }


class Choice(Hyperparameter):
    """A choice hyperparemeter for optimisation."""

    def __init__(self, value: float or int or bool or str,
                 values: list[float or int or bool or str],
                 is_ordered: bool = True, sort_values: bool = False):
        """Construct a new choice hyperparemeter."""
        if value not in values:
            raise ValueError('Provided value not in choices.')
        self.value, self.values, self.is_ordered, self.sort_values = (
            value, values, is_ordered, sort_values)

    def to_dict(self, name: str) -> dict:
        """Return a dict representation for use with Ax."""
        return {"name": name, "type": "choice", "values": self.values,
                "is_ordered": self.is_ordered, "sort_values": self.sort_values}


class Fixed(Hyperparameter):
    """A fixed hyperparemeter."""

    def __init__(self, value: float or int or bool or str):
        """Construct a new fixed hyperparemeter."""
        self.value = value

    def to_dict(self, name: str) -> dict:
        """Return a dict representation for use with Ax."""
        return {"name": name, "type": "fixed", "value": self.value}
