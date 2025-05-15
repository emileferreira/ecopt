from codecarbon import OfflineEmissionsTracker as Tracker
import mlflow

from .model import Model


class Meter:
    """An abstract energy consumption meter."""

    def __init__(self, experiment_name: str = None,
                 mlflow_tracking_uri: str = None):
        """
        Create a new meter.

        :param experiment_name: The experiment name for MLflow
        :param mlflow_tracking_uri: The MLflow tracking URI (defaults to local)
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

    def __call__(self, model: Model, utility_measure: str = "accuracy",
                 run_name: str = None, run_tags: dict = None,
                 skip_train: bool = False) -> dict:
        """Train and evaluate the model, returning an observation."""
        with mlflow.start_run(run_name=run_name):
            if run_tags is not None:
                mlflow.set_tags(run_tags)
            mlflow.log_params({
                name: hyperparameter.value for name, hyperparameter
                in model.hyperparameters.items()
            })
            metrics = self.observe(model, utility_measure, skip_train)
            mlflow.log_metrics(metrics)
        return metrics

    def observe(self, model: Model, utility_measure: str,
                skip_train: bool) -> dict:
        """Train and evaluate the model, returning an dict of measurements."""
        raise NotImplementedError


class CodeCarbonMeter(Meter):
    """An energy consumption meter using CodeCarbon."""

    def __init__(self, experiment_name: str = None,
                 mlflow_tracking_uri: str = None,
                 log_level: str = "INFO",
                 country_iso_code: str = "GBR",
                 default_cpu_power: float = None,
                 pue: float = None,
                 measure_power_secs: int = 15,
                 tracking_mode: str = "machine"):
        """
        Create a new meter.

        :param experiment_name: The experiment name for MLflow
        :param mlflow_tracking_uri: The MLflow tracking URI (defaults to local)
        :param log_level: The CodeCarbon log level ("debug", "info", "warning",
                          "error" or "critical")
        :param country_iso_code: The 3-letter ISO code of country location for
                           calculating carbon emissions
        :param default_cpu_power: The CPU power to be used if it is not known
        :param measure_power_secs: Interval in seconds to measure power usage
        :param tracking_mode: Either "process" or "machine" to measure the
                            energy consumption of the entire machine or attempt
                            to isolate the process
        """
        super().__init__(experiment_name, mlflow_tracking_uri)
        self.tracker_kwargs = {
            "country_iso_code": country_iso_code,
            "save_to_file": False,
            "log_level": log_level,
            "default_cpu_power": default_cpu_power,
            "pue": pue,
            "measure_power_secs": measure_power_secs,
            "tracking_mode": tracking_mode
        }

    def observe(self, model: Model, utility_measure: str,
                skip_train: bool) -> dict:
        """Train and evaluate the model, returning Metrics."""
        metrics = {}
        model.define()
        if not skip_train:
            with Tracker(**self.tracker_kwargs) as train_tracker:
                model.train()
            train_data = train_tracker.final_emissions_data
            metrics |= {
                "train_energy": train_data.energy_consumed * 1000,
                "train_carbon": train_data.emissions,
                "train_time": train_data.duration,
            }
        with Tracker(**self.tracker_kwargs) as evaluate_tracker:
            utility, num_samples = model.evaluate()
        evaluate_data = evaluate_tracker.final_emissions_data
        return metrics | {
            utility_measure: utility,
            "evaluate_energy": evaluate_data.energy_consumed * 1000,
            "evaluate_carbon": evaluate_data.emissions,
            "evaluate_time": evaluate_data.duration,
            "samples_per_wh": num_samples /
            (evaluate_data.energy_consumed * 1000),
            "samples_per_j": (num_samples /
                              (evaluate_data.energy_consumed * 1000)) / 3600,
            "samples_per_kg": num_samples / evaluate_data.emissions,
            "samples_per_s": num_samples / evaluate_data.duration
        }
