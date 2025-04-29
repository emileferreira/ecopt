from os import environ

import torchvision

from ecopt.meter import CodeCarbonMeter
from models.nn import NeuralNetworkModel
from ecopt.hyperparameter import Fixed

# load dataset
dataset_kwargs = {
    "root": "./data",
    "download": True,
    "transform": torchvision.transforms.ToTensor()
}
train_dataset = torchvision.datasets.MNIST(
    **dataset_kwargs, train=True)
eval_dataset = torchvision.datasets.MNIST(
    **dataset_kwargs, train=False)

# configure MLflow tracking
mlflow_tracking_uri = "https://mlflow.emileferreira.com"
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": False,
    "model": "nn",
    "dataset": "mnist"
}

meter = CodeCarbonMeter(
    experiment_name="Metrics",
    mlflow_tracking_uri=mlflow_tracking_uri)
for depth in range(1, 31):
    model = NeuralNetworkModel(train_dataset=train_dataset,
                               eval_dataset=eval_dataset,
                               depth=Fixed(depth),
                               hidden_size=Fixed(28 * 28 * 10))
    meter(model, utility_measure="weighted_f1",
          run_tags=run_tags, skip_train=True)
