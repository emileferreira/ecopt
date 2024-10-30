class Model:
    """A class to be extended by the user to adapt their model."""

    def __init__(self, hyperparams: dict):
        """Instantiate a model adapter, with the hyperparameters for
        optimising."""
        self.hyperparams = hyperparams

    def construct(self):
        """Construct the model using self.hyperparams and static
        hyperparameters."""
        raise NotImplementedError

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def evaluate(self) -> (float, int):
        """Evaluate the model, returning the accuracy and number of samples."""
        raise NotImplementedError
