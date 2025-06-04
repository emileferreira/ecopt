from os import environ

import torchvision

from ecopt.meter import CodeCarbonMeter
from models.nn import NeuralNetworkModel
from models.cnn import CNNModel
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
mlflow_tracking_uri = None

# NN
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": False,
    "model": "nn",
    "dataset": "mnist"
}
meter = CodeCarbonMeter(
    experiment_name="Metrics-depth",
    mlflow_tracking_uri=mlflow_tracking_uri)
for depth in range(1, 31):
    model = NeuralNetworkModel(train_dataset=train_dataset,
                               eval_dataset=eval_dataset,
                               depth=Fixed(depth),
                               hidden_size=Fixed(28 * 28 * 10))
    meter(model, utility_measure="weighted_f1",
          run_tags=run_tags, skip_train=True)

# CNN
run_tags["model"] = "cnn"
meter = CodeCarbonMeter(
    experiment_name="Metrics-stride",
    mlflow_tracking_uri=mlflow_tracking_uri)
for stride in range(2, 8):
    model = CNNModel(train_dataset=train_dataset,
                     eval_dataset=eval_dataset,
                     stride=Fixed(stride),
                     width=Fixed(256),
                     batch_size=Fixed(1000))
    meter(model, utility_measure="weighted_f1",
          run_tags=run_tags, skip_train=True)
