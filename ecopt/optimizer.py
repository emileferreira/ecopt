import torch
from codecarbon import OfflineEmissionsTracker as Tracker

from .model import Model

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class Optimizer:
    """An Energy Consumption Optimiser."""

    def __init__(self, model: Model, log_level="INFO", country_iso_code="GBR"):
        """Construct an Energy Consumption Optimiser for the provided model."""
        self.model = model
        self.tracker_kwargs = {
            "country_iso_code": country_iso_code,
            "save_to_file": False,
            "log_level": log_level
        }

    def measure(self) -> (float, float, float):
        """Construct, train and evaluate the model, returning the training
        energy in Wh, evaluate inferences per Wh and accuracy."""
        self.model.construct()
        with Tracker(**self.tracker_kwargs) as train_tracker:
            self.model.train()
        with Tracker(**self.tracker_kwargs) as evaluate_tracker:
            accuracy, num_samples = self.model.evaluate()
        train_energy = train_tracker.final_emissions_data\
            .energy_consumed * 1000
        evaluate_energy = evaluate_tracker.final_emissions_data\
            .energy_consumed * 1000
        inference_efficiency = num_samples / evaluate_energy
        return train_energy, inference_efficiency, accuracy

    def optimize(self, num_iterations):
        """Use Bayesian optimisation to tune self.model.hyperparams using
        num_iterations samples."""
        for i in range(num_iterations):
            _, inference_efficiency, accuracy = self.measure()
            # TODO: perform BO and update self.model.hyperparams
