import torch

from .model import Model

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class Optimizer:
    """An Energy Consumption Optimiser."""

    def __init__(self, model: Model):
        """Construct an optimiser for the provided model."""
        self.model = model

    # def optimize(self, )
