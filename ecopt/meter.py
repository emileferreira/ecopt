from codecarbon import OfflineEmissionsTracker as Tracker
import mlflow

from .model import Model


class Meter:
    """An abstract energy consumption meter."""

    def __init__(self, experiment_name: str = None,
                 mlflow_tracking_uri: str = None):
        """Create a new meter with an optional MLflow tracking URI."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

    def __call__(self, model: Model, run_name: str = None,
                 run_tags: dict = None) -> dict:
        """Train and evaluate the model, returning an observation."""
        with mlflow.start_run(run_name=run_name):
            if run_tags is not None:
                mlflow.set_tags(run_tags)
            metrics = self.observe(model)
            mlflow.log_params({
                name: hyperparameter.value for name, hyperparameter
                in model.hyperparameters.items()
            })
            mlflow.log_metrics(metrics)
        return metrics

    def observe(self, model: Model) -> dict:
        """Train and evaluate the model, returning an dict of measurements."""
        raise NotImplementedError


class CodeCarbonMeter(Meter):

    def __init__(self, experiment_name: str = None,
                 mlflow_tracking_uri: str = None,
                 log_level: str = "INFO",
                 country_iso_code: str = "GBR"):
        """Create a new meter."""
        super().__init__(experiment_name, mlflow_tracking_uri)
        self.tracker_kwargs = {
            "country_iso_code": country_iso_code,
            "save_to_file": False,
            "log_level": log_level
        }

    def observe(self, model: Model) -> dict:
        """Train and evaluate the model, returning Metrics."""
        with Tracker(**self.tracker_kwargs) as train_tracker:
            model.train()
        with Tracker(**self.tracker_kwargs) as evaluate_tracker:
            utility, num_samples = model.evaluate()
        train_data = train_tracker.final_emissions_data
        evaluate_data = evaluate_tracker.final_emissions_data
        metrics = {
            "utility": utility,
            "train_energy": train_data.energy_consumed * 1000,
            "train_carbon": train_data.emissions,
            "train_time": train_data.duration,
            "evaluate_energy": evaluate_data.energy_consumed * 1000,
            "evaluate_carbon": evaluate_data.emissions,
            "evaluate_time": evaluate_data.duration,
            "energy_efficiency": num_samples /
            evaluate_data.energy_consumed * 1000,
            "carbon_efficiency": num_samples / evaluate_data.emissions,
            "time_efficiency": num_samples / evaluate_data.duration
        }
        return metrics
