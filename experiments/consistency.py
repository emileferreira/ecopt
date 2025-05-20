from os import environ

from codecarbon.core.cpu import is_rapl_available
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed

from models.transformer import TextGenerationModel

mlflow_tracking_uri = "https://mlflow.emileferreira.com"
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": is_rapl_available(),
    "model": "textgeneration",
}
meter = CodeCarbonMeter(
    experiment_name="Consistency",
    mlflow_tracking_uri=mlflow_tracking_uri)
model = TextGenerationModel(model_name=Fixed("Qwen/Qwen2.5-0.5B"))
meter(model, run_tags=run_tags, skip_train=True)
