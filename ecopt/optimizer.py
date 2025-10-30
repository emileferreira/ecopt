from ax.plot.pareto_frontier import interact_pareto_frontier
from ax.plot.pareto_utils import get_observed_pareto_frontiers
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import render

from .model import Model
from .meter import Meter


class Optimizer:
    """The ECOpt optimiser."""

    def __init__(self, model: Model, meter: Meter,
                 performance_measure: str = "accuracy",
                 minimize_performance: bool = False,
                 efficiency_measure: str = "samples_per_j",
                 minimize_efficiency: bool = False,
                 skip_train: bool = False):
        """
        Construct an Energy Consumption Optimiser for the provided model.

        :param model: The model to optimise
        :param meter: The meter to use in optimising the model
        :param performance_measure: The name of the user-defined performance metric
        :param minimize_performance: Whether or not to minimise the performance measure
        :param efficiency_measure: The name of the efficiency metric for which
                                   to optimise
        :param minimize_efficiency: Whether or not to minimise the efficiency
                                    measure
        :param skip_train: Whether or not to skip the call to `model.train`
        """
        self.model = model
        self.meter = meter
        self.performance_measure = performance_measure
        self.minimize_performance = minimize_performance
        self.efficiency_measure = efficiency_measure
        self.minimize_efficiency = minimize_efficiency
        self.skip_train = skip_train
        self.ax_client = AxClient()

    def __call__(self, num_init_steps: int = 5, num_opt_steps: int = 20,
                 performance_threshold: float = None,
                 efficiency_threshold: float = None,
                 run_tags: dict = None):
        """
        Use Bayesian optimisation to tune the model hyperparameters.

        :param num_init_steps: The number of Sobol' points to sample before
                               optimising
        :param num_opt_steps: The number of multi-objective Bayesian
                              optimisation (MOBO) points to sample
        :param performance_threshold: A MOBO threshold value the performance metric
        :param efficiency_threshold: A MOBO threshold value for efficiency
        :param run_tags: The tags for the MLflow run
        """
        self.ax_client.create_experiment(
            parameters=[hyperparameter.to_dict(name) for name, hyperparameter
                        in self.model.hyperparameters.items()],
            objectives={
                self.performance_measure: ObjectiveProperties(
                    minimize=self.minimize_performance,
                    threshold=performance_threshold),
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
            metrics = self.meter(self.model, self.performance_measure,
                                 run_tags=run_tags, skip_train=self.skip_train)
            raw_data = {
                self.performance_measure: metrics[self.performance_measure],
                self.efficiency_measure: metrics[self.efficiency_measure]
            }
            self.ax_client.complete_trial(trial_index=trial_index,
                                          raw_data=raw_data)

    def plot_pareto_frontier(self, CI_level: float = 0.90):
        """
        Plot the Pareto frontier of the observations.

        :param CI_level: The confidence intervals to include in the plot
        """
        experiment = self.ax_client.experiment
        frontier = get_observed_pareto_frontiers(experiment, rel=False)
        render(interact_pareto_frontier(frontier, CI_level=CI_level))
