from .hyperparameter import Hyperparameter


class Model:
    """A model adapter for ECOpt, to be extended by the user."""

    def __init__(self):
        """Instantiate a model adapter and define the hyperparameters for
        optimisation as instance variables."""
        raise NotImplementedError

    def define(self):
        """Construct the model using the (potentially updated)
        hyperparameters."""
        pass

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def evaluate(self) -> (float, int):
        """Evaluate the model, returning the utility and number of
        inferences."""
        raise NotImplementedError

    @property
    def hyperparameters(self) -> dict:
        """The hyperparameter instance variables."""
        return {key: value for key, value in vars(self).items()
                if isinstance(value, Hyperparameter)}
