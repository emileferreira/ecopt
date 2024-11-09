from .hyperparam import Hyperparam


class Model:
    """A model adapter for ECOpt, to be extended by the user."""

    def __init__(self):
        """Instantiate a model adapter and define the hyperparameters for
        optimision as instance variables."""
        raise NotImplementedError

    def train(self):
        """Construct and train the model using the optimised and static
        hyperparameters."""
        raise NotImplementedError

    def evaluate(self) -> (float, int):
        """Evaluate the model, returning the utility (greater is better) and
        number of inferences."""
        raise NotImplementedError

    @property
    def hyperparams(self) -> dict:
        """The hyperparameter instance variables."""
        return {key: value for key, value in vars(self).items()
                if isinstance(value, Hyperparam)}
