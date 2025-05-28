from os import environ

from codecarbon.core.cpu import is_rapl_available
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed, Range, Choice
from ecopt.optimizer import Optimizer
import torchvision

from models.cnn import CNNModel

# load dataset
dataset_kwargs = {
    "root": "./data",
    "download": True,
    "transform": torchvision.transforms.ToTensor()
}
train_dataset = torchvision.datasets.CIFAR10(
    **dataset_kwargs, train=True)
eval_dataset = torchvision.datasets.CIFAR10(
    **dataset_kwargs, train=False)

mlflow_tracking_uri = "https://mlflow.emileferreira.com"
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": is_rapl_available()
}
meter = CodeCarbonMeter(
    experiment_name="CNN",
    mlflow_tracking_uri=mlflow_tracking_uri)
model = CNNModel(train_dataset=train_dataset,
                 eval_dataset=eval_dataset,
                 batch_size=Fixed(64),
                 num_epochs=Fixed(50),
                 learning_rate=Fixed(0.001),
                 output_size=Fixed(10),
                 width=Range(128, min=1, max=128),
                 depth=Range(6, min=1, max=6),
                 input_width=Fixed(32),
                 input_height=Fixed(32),
                 input_channels=Fixed(3),
                 kernel_size=Choice(
                     9, values=[1, 3, 5, 7, 9], is_ordered=True),
                 stride=Fixed(1),
                 padding=Fixed(-1),
                 pool=Choice(False, values=[True, False], is_ordered=False))
optimizer = Optimizer(model, meter, utility_measure="weighted_f1")
optimizer(num_init_steps=5, num_opt_steps=95, run_tags=run_tags)
