from os import environ

from codecarbon.core.cpu import is_rapl_available
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed

from models.transformer import TextGenerationModel

mlflow_tracking_uri = "https://mlflow.emileferreira.com"
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": is_rapl_available()
}
model_names = [
    "openai-community/gpt2",  # 132M, 2019
    "Qwen/Qwen3-0.6B",  # 752M, 2025
    "google/gemma-3-4b-pt",  # 4B, 2025
    "meta-llama/Llama-3.1-8B",  # 8.03B, 2024
]
meter = CodeCarbonMeter(
    experiment_name="Consistency",
    mlflow_tracking_uri=mlflow_tracking_uri)
for model_name in model_names:
    model = TextGenerationModel(model_name=Fixed(model_name))
    meter(model, run_tags=run_tags, skip_train=True)
