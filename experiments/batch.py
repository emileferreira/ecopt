from os import environ

from codecarbon.core.cpu import is_rapl_available
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed, Range
from ecopt.optimizer import Optimizer

from models.transformer import TextGenerationModel

mlflow_tracking_uri = "https://mlflow.emileferreira.com"
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": is_rapl_available()
}
meter = CodeCarbonMeter(
    experiment_name="Batch",
    mlflow_tracking_uri=mlflow_tracking_uri)
model = TextGenerationModel(model_name=Fixed("google/gemma-3-4b-pt"),
                            batch_size=Range(1, min=1, max=2000),
                            num_inferences=Fixed(2000))
optimizer = Optimizer(model, meter, skip_train=True)
optimizer(num_init_steps=5, num_opt_steps=25, run_tags=run_tags)
