from ax.plot.pareto_frontier import interact_pareto_frontier
from ax.plot.pareto_utils import get_observed_pareto_frontiers
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import render

from .model import Model
from .meter import Meter


class Optimizer:
    """An Energy Consumption Optimiser."""

    def __init__(self, model: Model, meter: Meter):
        """Construct an Energy Consumption Optimiser for the provided model."""
        self.model = model
        self.meter = meter
        self.observations = []
        self.ax_client = AxClient()

    def __call__(self, num_init_steps: int = 5, num_opt_steps: int = 20):
        """Use Bayesian optimisation to tune hyperparameters using
        num_iterations observations."""
        self.ax_client.create_experiment(
            parameters=[hyperparameter.to_dict(name) for name, hyperparameter
                        in self.model.hyperparameters.items()],
            objectives={
                "utility": ObjectiveProperties(minimize=False),
                "energy_efficiency": ObjectiveProperties(minimize=False),
            },
            choose_generation_strategy_kwargs={
                "num_initialization_trials": num_init_steps,
            }
        )
        for _ in range(num_init_steps + num_opt_steps):
            parameters, trial_index = self.ax_client.get_next_trial()
            for key, value in parameters.items():
                self.model.hyperparameters[key].value = value
            observation = self.meter(self.model)
            self.observations.append(observation)
            raw_data = {
                "utility": (observation.utility, 0.0),
                "energy_efficiency": (observation.energy_efficiency, 0.0)
            }
            self.ax_client.complete_trial(trial_index=trial_index,
                                          raw_data=raw_data)

    def plot_pareto_frontier(self, CI_level: float = 0.90):
        """Plot the Pareto frontier of the observations."""
        experiment = self.ax_client.experiment
        frontier = get_observed_pareto_frontiers(experiment, rel=False)
        render(interact_pareto_frontier(frontier, CI_level=CI_level))
