from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import render

from .model import Model
from .meter import Meter
from .hyperparameter import Range, Choice, Fixed


class Optimizer:
    """An Energy Consumption Optimiser."""

    def __init__(self, model: Model, meter: Meter):
        """Construct an Energy Consumption Optimiser for the provided model."""
        self.model = model
        self.meter = meter
        self.observations = []
        self.ax_client = AxClient()

    def __call__(self, num_iterations):
        """Use Bayesian optimisation to tune hyperparameters using
        num_iterations observations."""
        parameters = []
        for name, hyperparameter in self.model.hyperparameters.items():
            parameter = {"name": name}
            if type(hyperparameter) is Range:
                parameter["type"] = "range"
                parameter["bounds"] = [hyperparameter.min, hyperparameter.max]
                parameter["log_scale"] = hyperparameter.log_scale
            elif type(hyperparameter) is Choice:
                parameter["type"] = "choice"
                parameter["values"] = hyperparameter.values
            elif type(hyperparameter) is Fixed:
                parameter["type"] = "fixed"
                parameter["value"] = hyperparameter.value
            else:
                raise ValueError(f"Unexpected type: {type(hyperparameter)}")
            parameters.append(parameter)
        self.ax_client.create_experiment(
            parameters=parameters, objectives={
                "utility": ObjectiveProperties(minimize=False),
                "energy_efficiency": ObjectiveProperties(minimize=False),
            }
        )
        for _ in range(num_iterations):
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

    def plot_pareto_frontier(self, num_points=20, CI_level=0.90):
        """Plot the Pareto frontier."""
        experiment = self.ax_client.experiment
        objectives = experiment.optimization_config.objective.objectives
        frontier = compute_posterior_pareto_frontier(
            experiment=self.ax_client.experiment,
            data=self.ax_client.experiment.fetch_data(),
            primary_objective=objectives[1].metric,
            secondary_objective=objectives[0].metric,
            absolute_metrics=["utility", "energy_efficiency"],
            num_points=num_points,
        )
        render(plot_pareto_frontier(frontier, CI_level=CI_level))
