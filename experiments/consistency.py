from os import environ

from ecopt.meter import CodeCarbonMeter
from models.vit import ViTMNISTModel

mlflow_tracking_uri = "https://mlflow.emileferreira.com"
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": False,
    "model": "vit",
    "dataset": "mnist"
}
meter = CodeCarbonMeter(
    experiment_name="Consistency",
    mlflow_tracking_uri=mlflow_tracking_uri)
model = ViTMNISTModel()
meter(model, utility_measure="weighted_f1",
      run_tags=run_tags, skip_train=True)
