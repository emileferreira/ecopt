from time import time
from os import remove

from codecarbon import OfflineEmissionsTracker as Tracker
from eco2ai import Tracker as eco2AITracker

from .model import Model


class Observation:
    """A container for metrics."""

    def __init__(self, hyperparameters: dict, utility: float,
                 num_inferences: int, train_energy: float, train_carbon: float,
                 train_time: float, evaluate_energy: float,
                 evaluate_carbon: float, evaluate_time: float):
        """Construct a new observation, provided energy metrics in Wh, carbon
        metrics in kg and time metrics in seconds."""
        self.hyperparameters = hyperparameters
        self.utility = utility
        self.train_energy = train_energy
        self.train_carbon = train_carbon
        self.train_time = train_time
        self.energy_efficiency = num_inferences / evaluate_energy
        self.carbon_efficiency = num_inferences / evaluate_carbon
        self.time_efficiency = num_inferences / evaluate_time

    def __str__(self):
        """Return a str representation of the observation"""
        return str(vars(self))


class Meter:
    """An abstract energy consumption meter."""

    def __init__(self):
        """Create a new meter."""
        raise NotImplementedError

    def __call__(self, model: Model) -> Observation:
        """Train and evaluate the model, returning an observation."""
        raise NotImplementedError


class CodeCarbonMeter(Meter):

    def __init__(self, log_level: str = "INFO", country_iso_code: str = "GBR"):
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
            utility, num_inferences = model.evaluate()
        hyperparameters = {name: hyperparameter.value for name, hyperparameter
                           in model.hyperparameters.items()}
        return Observation(
            hyperparameters, utility, num_inferences,
            train_tracker.final_emissions_data.energy_consumed * 1000,
            train_tracker.final_emissions_data.emissions,
            train_tracker.final_emissions_data.duration,
            evaluate_tracker.final_emissions_data.energy_consumed * 1000,
            evaluate_tracker.final_emissions_data.emissions,
            evaluate_tracker.final_emissions_data.duration
        )


class eco2AIMeter(Meter):

    def __init__(self, measure_period: float = 1,
                 country_alpha_2_code: str = "GB",
                 cpu_processes: str = "current"):
        """Create a new meter."""
        self.tracker_kwargs = {
            "measure_period": measure_period,
            "alpha_2_code": country_alpha_2_code,
            "cpu_processes": cpu_processes
        }

    def __call__(self, model: Model) -> Observation:
        """Train and evaluate the model, returning Metrics."""

        def emissions(consumption: float, emissions_level: float) -> float:
            """Calculate the emissions in kg."""
            return consumption * emissions_level / 1000

        train_tracker = eco2AITracker(**self.tracker_kwargs)
        train_tracker.start()
        start_time = time()
        model.train()
        train_time = time() - start_time
        train_energy = train_tracker.consumption() * 1000
        train_carbon = emissions(train_tracker.consumption(),
                                 train_tracker.emission_level())
        train_tracker.stop()
        evaluate_tracker = eco2AITracker(**self.tracker_kwargs)
        evaluate_tracker.start()
        start_time = time()
        utility, num_inferences = model.evaluate()
        evaluate_time = time() - start_time
        evaluate_energy = evaluate_tracker.consumption() * 1000
        evaluate_carbon = emissions(evaluate_tracker.consumption(),
                                    evaluate_tracker.emission_level())
        evaluate_tracker.stop()
        try:
            remove("my_emission_file.csv")
        except OSError:
            pass
        hyperparameters = {name: hyperparameter.value for name, hyperparameter
                           in model.hyperparameters.items()}
        return Observation(hyperparameters, utility, num_inferences,
                           train_energy, train_carbon, train_time,
                           evaluate_energy, evaluate_carbon, evaluate_time)
