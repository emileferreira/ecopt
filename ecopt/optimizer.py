from ax.plot.pareto_frontier import interact_pareto_frontier
from ax.plot.pareto_utils import get_observed_pareto_frontiers
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import render

from .model import Model
from .meter import Meter


class Optimizer:
    """An Energy Consumption Optimiser."""

    def __init__(self, model: Model, meter: Meter,
                 utility_measure: str = "accuracy",
                 minimize_utility: bool = False,
                 efficiency_measure: str = "samples_per_wh",
                 minimize_efficiency: bool = False):
        """Construct an Energy Consumption Optimiser for the provided model."""
        self.model = model
        self.meter = meter
        self.utility_measure = utility_measure
        self.minimize_utility = minimize_utility
        self.efficiency_measure = efficiency_measure
        self.minimize_efficiency = minimize_efficiency
        self.ax_client = AxClient()

    def __call__(self, num_init_steps: int = 5, num_opt_steps: int = 20,
                 utility_threshold: float = None,
                 efficiency_threshold: float = None,
                 run_tags: dict = None):
        """Use Bayesian optimisation to tune hyperparameters using
        num_iterations observations."""
        self.ax_client.create_experiment(
            parameters=[hyperparameter.to_dict(name) for name, hyperparameter
                        in self.model.hyperparameters.items()],
            objectives={
                self.utility_measure: ObjectiveProperties(
                    minimize=self.minimize_utility,
                    threshold=utility_threshold),
                self.efficiency_measure: ObjectiveProperties(
                    minimize=self.minimize_efficiency,
                    threshold=efficiency_threshold),
            },
            choose_generation_strategy_kwargs={
                "num_initialization_trials": num_init_steps,
            }
        )
        for _ in range(num_init_steps + num_opt_steps):
            parameters, trial_index = self.ax_client.get_next_trial()
            for key, value in parameters.items():
                self.model.hyperparameters[key].value = value
            metrics = self.meter(self.model, self.utility_measure,
                                 run_tags=run_tags)
            raw_data = {
                self.utility_measure: metrics[self.utility_measure],
                self.efficiency_measure: metrics[self.efficiency_measure]
            }
            self.ax_client.complete_trial(trial_index=trial_index,
                                          raw_data=raw_data)

    def plot_pareto_frontier(self, CI_level: float = 0.90):
        """Plot the Pareto frontier of the observations."""
        experiment = self.ax_client.experiment
        frontier = get_observed_pareto_frontiers(experiment, rel=False)
        render(interact_pareto_frontier(frontier, CI_level=CI_level))
