from os import environ

import torchvision
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed
from codecarbon.core.cpu import is_rapl_available

from models.cnn import LeNet5Model

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
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": is_rapl_available()
}

# measure distribution of metrics
meter = CodeCarbonMeter(
    experiment_name="RAPL-constant",
    mlflow_tracking_uri=mlflow_tracking_uri)
for _ in range(35):
    model = LeNet5Model(train_dataset=train_dataset,
                        eval_dataset=eval_dataset)
    meter(model, utility_measure="weighted_f1", run_tags=run_tags)

# explore sclaing the batch size
meter = CodeCarbonMeter(
    experiment_name="RAPL-batch",
    mlflow_tracking_uri=mlflow_tracking_uri)
for exp in range(4, 14):
    batch_size = Fixed(2**exp)
    model = LeNet5Model(train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        batch_size=batch_size)
    meter(model, utility_measure="weighted_f1", run_tags=run_tags)

# explore sclaing the number of epochs
meter = CodeCarbonMeter(
    experiment_name="RAPL-epoch",
    mlflow_tracking_uri=mlflow_tracking_uri)
for num_epochs in range(1, 11):
    model = LeNet5Model(train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        num_epochs=Fixed(num_epochs))
    meter(model, utility_measure="weighted_f1", run_tags=run_tags)
