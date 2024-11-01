from codecarbon import OfflineEmissionsTracker as Tracker

from .model import Model


class Metrics:
    """A container for metrics."""

    def __init__(self, train_energy, train_carbon, evaluate_energy,
                 evaluate_carbon, accuracy, num_samples):
        """Construct a new container, provided energy metrics in Wh and carbon
        metrics in kg."""
        self.train_energy = train_energy
        self.train_carbon = train_carbon
        self.energy_efficiency = num_samples / evaluate_energy
        self.carbon_efficiency = num_samples / evaluate_carbon
        self.accuracy = accuracy


class Meter:

    def __init__(self, log_level, country_iso_code):
        self.tracker_kwargs = {
            "country_iso_code": country_iso_code,
            "save_to_file": False,
            "log_level": log_level
        }

    def __call__(self, model: Model) -> Metrics:
        """Train and evaluate the model, returning Metrics."""
        with Tracker(**self.tracker_kwargs) as train_tracker:
            model.train()
        with Tracker(**self.tracker_kwargs) as evaluate_tracker:
            accuracy, num_samples = model.evaluate()
        return Metrics(
            train_tracker.final_emissions_data.energy_consumed * 1000,
            train_tracker.final_emissions_data.emissions,
            evaluate_tracker.final_emissions_data.energy_consumed * 1000,
            evaluate_tracker.final_emissions_data.emissions,
            accuracy, num_samples
        )
