class Hyperparam:
    """A hyperparemeter for optimisation."""

    def __init__(self, value: float or int,
                 min: float or int = None,
                 max: float or int = None):
        """Construct a new hyperparemeter."""
        self.value, self.min, self.max = value, min, max
