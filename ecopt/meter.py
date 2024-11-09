from codecarbon import OfflineEmissionsTracker as Tracker

from .model import Model


class Observation:
    """A container for metrics."""

    def __init__(self, hyperparams: dict, utility: float, num_samples: int,
                 train_energy: float, train_carbon: float, train_time: float,
                 evaluate_energy: float, evaluate_carbon: float,
                 evaluate_time: float):
        """Construct a new observation, provided energy metrics in Wh, carbon
        metrics in kg and time metrics in seconds."""
        self.hyperparams = hyperparams
        self.utility = utility
        self.train_energy = train_energy
        self.train_carbon = train_carbon
        self.train_time = train_time
        self.energy_efficiency = num_samples / evaluate_energy
        self.carbon_efficiency = num_samples / evaluate_carbon
        self.time_efficiency = num_samples / evaluate_time


class Meter:

    def __init__(self, log_level: str, country_iso_code: str):
        """Create a new meter."""
        self.tracker_kwargs = {
            "country_iso_code": country_iso_code,
            "save_to_file": False,
            "log_level": log_level
        }

    def __call__(self, model: Model) -> Observation:
        """Train and evaluate the model, returning Metrics."""
        with Tracker(**self.tracker_kwargs) as train_tracker:
            model.train()
        with Tracker(**self.tracker_kwargs) as evaluate_tracker:
            utility, num_samples = model.evaluate()
        hyperparams = {key: hyperparam.value for key, hyperparam
                       in model.hyperparams.items()}
        return Observation(
            hyperparams, utility, num_samples,
            train_tracker.final_emissions_data.energy_consumed * 1000,
            train_tracker.final_emissions_data.emissions,
            train_tracker.final_emissions_data.duration,
            evaluate_tracker.final_emissions_data.energy_consumed * 1000,
            evaluate_tracker.final_emissions_data.emissions,
            evaluate_tracker.final_emissions_data.duration
        )
