from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import mlflow
from thop import profile

from ecopt.model import Model
from ecopt.hyperparameter import Fixed, Hyperparameter


class LeNet5(torch.nn.Module):
    """LetNet-5 model."""

    def __init__(self, num_classes=10):
        """Instantiate a new model."""
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """Apply the model to a 28x28 grayscale input."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet5Model(Model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_dataset: Dataset, eval_dataset: Dataset,
                 batch_size: Hyperparameter = Fixed(100),
                 num_epochs: Hyperparameter = Fixed(10),
                 learning_rate: Hyperparameter = Fixed(0.001),
                 output_size: Hyperparameter = Fixed(10),
                 utility_measure: str = "weighted_f1"):
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_size = output_size
        has_gpus = torch.cuda.device_count() > 0
        self.dataloader_kwargs = {
            "num_workers": min(cpu_count(), 32) if has_gpus else 0,
            "pin_memory": has_gpus,
            "prefetch_factor": 2 if has_gpus else None,
            "persistent_workers": has_gpus
        }
        self.utility_measure = utility_measure

    def define(self):
        self.model = LeNet5(num_classes=self.output_size.value).to(self.device)

    def train(self):
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=self.batch_size.value,
                                shuffle=True, **self.dataloader_kwargs)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate.value)
        epoch_losses = []
        with tqdm(total=self.num_epochs.value, unit="epoch",
                  desc="Train") as progress:
            for epoch in range(self.num_epochs.value):
                epoch_loss = 0.0
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    epoch_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1 / len(dataloader))
                epoch_losses.append(epoch_loss / len(dataloader))

    def evaluate(self) -> (float, int):
        image = self.eval_dataset[0][0]
        image = image.view(1, *image.size()).to(self.device)
        macs, parameters = profile(self.model, inputs=(image,))
        mlflow.log_metrics({
            "parameters": parameters,
            "flops": 2 * macs
        })
        dataloader = DataLoader(dataset=self.eval_dataset,
                                batch_size=self.batch_size.value,
                                shuffle=False, **self.dataloader_kwargs)
        if self.utility_measure == "accuracy":
            utility = self.evaluate_accuracy(dataloader)
        elif self.utility_measure == "weighted_f1":
            utility = self.evaluate_f1(dataloader)
        else:
            raise NotImplementedError
        num_samples = len(self.eval_dataset) + 1  # add one for the profiler
        return utility, num_samples

    def evaluate_f1(self, dataloader) -> float:
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(dataloader, unit="batch",
                                       desc="Evaluate"):
                images = images.to(self.device)
                y_true.append(labels)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                y_pred.append(predictions.cpu())
        y_true, y_pred = np.concat(y_true), np.concat(y_pred)
        return f1_score(y_true, y_pred, average="weighted")

    def evaluate_accuracy(self, dataloader) -> float:
        num_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, unit="batch",
                                       desc="Evaluate"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                num_correct += (predictions == labels).sum().item()
        return num_correct / len(self.eval_dataset)


class CNN(torch.nn.Module):

    def __init__(self, width: int, depth: int, input_width: int,
                 input_height: int, input_channels: int, kernel_size: int,
                 stride: int, padding: int, output_size: int, pool: bool):
        super(CNN, self).__init__()
        # hack to let padding adapt to kernel size
        if padding == -1:
            padding = kernel_size // 2
        layers = [torch.nn.Conv2d(
            input_channels, width, kernel_size, stride=stride, padding=padding
        ), torch.nn.ReLU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Conv2d(
                width, width, kernel_size, stride=stride, padding=padding
            ), torch.nn.ReLU()]
            if pool:
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        self.feature_extractor = torch.nn.Sequential(*layers)
        # automatically calculate classifier dimension
        with torch.no_grad():
            dummy_input = torch.zeros(
                2, input_channels, input_height, input_width)
            dummy_output = self.feature_extractor(dummy_input)
            _, c, h, w = dummy_output.shape
            flattened_size = c * h * w
        self.classifier = torch.nn.Linear(flattened_size, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class CNNModel(LeNet5Model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_dataset: Dataset, eval_dataset: Dataset,
                 batch_size: Hyperparameter = Fixed(100),
                 num_epochs: Hyperparameter = Fixed(10),
                 learning_rate: Hyperparameter = Fixed(0.001),
                 output_size: Hyperparameter = Fixed(10),
                 width: Hyperparameter = Fixed(128),
                 depth: Hyperparameter = Fixed(10),
                 input_width: Hyperparameter = Fixed(28),
                 input_height: Hyperparameter = Fixed(28),
                 input_channels: Hyperparameter = Fixed(1),
                 kernel_size: Hyperparameter = Fixed(3),
                 stride: Hyperparameter = Fixed(1),
                 padding: Hyperparameter = Fixed(1),
                 pool: Hyperparameter = Fixed(False),
                 utility_measure: str = "weighted_f1"):
        super().__init__(train_dataset, eval_dataset, batch_size, num_epochs,
                         learning_rate, output_size, utility_measure)
        self.width, self.depth = width, depth
        self.input_width, self.input_height = input_height, input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride, self.padding, self.pool = stride, padding, pool

    def define(self):
        self.model = CNN(self.width.value, self.depth.value,
                         self.input_width.value, self.input_height.value,
                         self.input_channels.value, self.kernel_size.value,
                         self.stride.value, self.padding.value,
                         self.output_size.value, self.pool.value
                         ).to(self.device)
