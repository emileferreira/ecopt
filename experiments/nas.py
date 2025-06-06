from os import environ

from codecarbon.core.cpu import is_rapl_available
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed, Range, Choice
from ecopt.optimizer import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from models.cnn import CNNModel

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])
dataset_kwargs = {
    "root": "./data",
    "download": True,
    "transform": transform
}
train_dataset = datasets.CIFAR10(
    **dataset_kwargs, train=True)
targets = np.array(train_dataset.targets)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)
eval_dataset = datasets.CIFAR10(
    **dataset_kwargs, train=False)

mlflow_tracking_uri = None
run_tags = {
    "machine": environ["ECOPT_MACHINE"],
    "rapl": is_rapl_available()
}
meter = CodeCarbonMeter(
    experiment_name="NAS",
    mlflow_tracking_uri=mlflow_tracking_uri)
utility_measure = "accuracy"
model = CNNModel(train_dataset=train_subset,
                 eval_dataset=eval_dataset,
                 val_dataset=val_subset,
                 batch_size=Fixed(64),
                 num_epochs=Fixed(500),
                 learning_rate=Fixed(0.001),
                 output_size=Fixed(10),
                 stop_early=Fixed(True),
                 patience=Fixed(3),
                 min_delta=Fixed(0.001),
                 width=Range(51, min=1, max=128),
                 depth=Range(6, min=1, max=6),
                 input_width=Fixed(32),
                 input_height=Fixed(32),
                 input_channels=Fixed(3),
                 kernel_size=Choice(
                     3, values=[1, 3, 5, 7, 9], is_ordered=True),
                 stride=Fixed(1),
                 padding=Fixed(-1),
                 pool=Choice(True, values=[True, False], is_ordered=True),
                 utility_measure=utility_measure)
optimizer = Optimizer(model, meter, utility_measure=utility_measure)
optimizer(num_init_steps=40, num_opt_steps=360,
          utility_threshold=0,
          efficiency_threshold=0,
          run_tags=run_tags)
