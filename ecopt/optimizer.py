import torch

from .model import Model
from .meter import Meter

torch_kwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class Optimizer:
    """An Energy Consumption Optimiser."""

    def __init__(self, model: Model, log_level: str = "INFO",
                 country_iso_code: str = "GBR"):
        """Construct an Energy Consumption Optimiser for the provided model."""
        self.model = model
        self.meter = Meter(log_level, country_iso_code)
        self.observations = []

    def __call__(self, num_iterations):
        """Use Bayesian optimisation to tune hyperparameters using
        num_iterations samples."""
        for i in range(num_iterations):
            observation = self.meter(self.model)
            self.observations.append(observation)
            print(f"Consumed {observation.train_energy} Wh during training")
            print(f"Achieved {observation.utility*100}% accuracy " +
                  f"at {observation.energy_efficiency} inferences/Wh " +
                  f"and {observation.time_efficiency} inferences/s")
            # TODO: perform BO and update hyperparameters
