class Model:
    """A model adapter for ECOpt, to be extended by the user."""

    def __init__(self, hyperparams: dict):
        """Instantiate a model adapter, with the hyperparameters for
        optimising."""
        self.hyperparams = hyperparams

    def train(self):
        """Construct and train the model using self.hyperparams and static
        hyperparameters."""
        raise NotImplementedError

    def evaluate(self) -> (float, int):
        """Evaluate the model, returning the accuracy and number of samples."""
        raise NotImplementedError
