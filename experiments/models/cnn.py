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


class EarlyStopping:
    """A class to enable early stopping of training. It signals to stop after
    the validation loss delta has been less than some threshold for some number
    of epochs."""

    def __init__(self, patience=10, min_delta=0):
        """
        Construct a new early stopping widget.

        :param patience: The number of epochs to wait for the loss delta to
                         improve
        :param min_delta: The minimum validation loss delta required to
                          continue training
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss) -> bool:
        """
        Return whether or not to stop training.

        :param val_loss: The validation loss of the current epoch
        :return: Whether or not to continue training
        """
        print(f"Val loss delta: {self.best_loss - val_loss}")
        if val_loss < self.best_loss - self.min_delta:
            self.counter = 0
            self.best_loss = val_loss
        else:
            self.counter += 1
        return self.counter >= self.patience


class LeNet5(torch.nn.Module):
    """LetNet-5 model, using ReLU activations and max pooling."""

    def __init__(self, num_classes=10):
        """
        Instantiate a new model.

        :param num_classes: The number of output classes
        """
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to a 28x28 grayscale input.

        :param x: The input image
        :return: The classification outputs
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet5Model(Model):
    """A model adapter for `LetNet-5`."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_dataset: Dataset, eval_dataset: Dataset,
                 val_dataset: Dataset = None,
                 batch_size: Hyperparameter = Fixed(100),
                 num_epochs: Hyperparameter = Fixed(10),
                 learning_rate: Hyperparameter = Fixed(0.001),
                 output_size: Hyperparameter = Fixed(10),
                 stop_early: Hyperparameter = Fixed(False),
                 patience: Hyperparameter = Fixed(5),
                 min_delta: Hyperparameter = Fixed(0.001),
                 performance_measure: str = "weighted_f1"):
        """
        Instantiate the model adapter and define the hyperparameters for
        optimisation as instance variables.

        :param train_dataset: The dataset used for training
        :param eval_dataset: The dataset used for evaluation
        :param val_dataset: The dataset used for valuation
        :param batch_size: The batch size to use for training and evaluation
        :param num_epochs: The number of epochs to train on
        :param learning_rate: The learning rate for the Adam optimiser
        :param output_size: The dimension of the output layer
        :param stop_early: Whether or not to use early stopping
        :param patience: The number of epochs to wait for the loss delta to
                         improve
        :param min_delta: The minimum validation loss delta required to
                          continue training
        :param performance_measure: The performance measure to use in evaluation
        """
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.stop_early = stop_early
        self.patience = patience
        self.min_delta = min_delta
        has_gpus = torch.cuda.device_count() > 0
        self.dataloader_kwargs = {
            "num_workers": min(cpu_count(), 32) if has_gpus else 0,
            "pin_memory": has_gpus,
            "prefetch_factor": 2 if has_gpus else None,
            "persistent_workers": has_gpus
        }
        self.performance_measure = performance_measure

    def define(self):
        """Construct the model using the (potentially updated)
        hyperparameters."""
        self.model = LeNet5(num_classes=self.output_size.value).to(self.device)

    def train(self):
        """Train the model."""
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size.value,
            shuffle=True, **self.dataloader_kwargs)
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size.value,
            shuffle=False, **self.dataloader_kwargs
        ) if self.val_dataset is not None else None
        if self.stop_early.value and val_dataloader is None:
            raise ValueError("Cannot stop early without validation set.")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate.value)
        train_losses, val_losses = [], []
        early_stopping = EarlyStopping(
            self.patience.value, self.min_delta.value)
        with tqdm(total=self.num_epochs.value, unit="epoch",
                  desc="Train") as progress:
            for epoch in range(self.num_epochs.value):
                train_loss = val_loss = 0.0
                self.model.train()
                for images, labels in train_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1 / len(train_dataloader))
                train_loss /= len(train_dataloader)
                print(f"Train loss: {train_loss}")
                train_losses.append(train_loss)
                if val_dataloader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for images, labels in val_dataloader:
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                    val_loss /= len(val_dataloader)
                    print(f"Val loss: {val_loss}")
                    val_losses.append(val_loss)
                    if self.stop_early.value and early_stopping(val_loss):
                        print("Stopping early.")
                        break
            epochs = len(train_losses)
            mlflow.log_metrics({"stopped_after": epochs})
            print("Train losses:")
            print(train_losses)
            print("Val losses:")
            print(val_losses)

    def evaluate(self) -> (float, int):
        """
        Evaluate the model.

        :return: The measured performance and the number of samples
        """
        self.model.eval()
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
        if self.performance_measure == "accuracy":
            performance = self.evaluate_accuracy(dataloader)
        elif self.performance_measure == "weighted_f1":
            performance = self.evaluate_f1(dataloader)
        else:
            raise NotImplementedError
        num_samples = len(self.eval_dataset) + 1  # add one for the profiler
        return performance, num_samples

    def evaluate_f1(self, dataloader) -> float:
        """
        Evaluate the F1 score of the model.

        :param dataloader: The dataloader of the evaluation dataset
        :return: The F1 score
        """
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
        """
        Evaluate the accuracy of the model.

        :param dataloader: The dataloader of the evaluation dataset
        :return: The accuracy score
        """
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
    """A parameterised convolutional neural network."""

    def __init__(self, width: int, depth: int, input_width: int,
                 input_height: int, input_channels: int, kernel_size: int,
                 stride: int, padding: int, output_size: int, pool: bool):
        """
        Construct a new CNN.

        :param width: The number of filters per convolutional layer
        :param depth: The number of convolutional layers
        :param input_width: The width of the input in pixels
        :param input_height: The height of the input in pixels
        :param input_channels: The number of input colour channels
        :param kernel_size: The size of the filters (kernel_size x kernel_size)
        :param stride: The stride of the filters
        :param padding: The conv. padding
        :param pool: Whether or not to use max pooling after each conv. layer
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input.

        :param x: The input image
        :return: The classification outputs
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class CNNModel(LeNet5Model):
    """A model adapter for `CNN`."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_dataset: Dataset, eval_dataset: Dataset,
                 val_dataset: Dataset = None,
                 batch_size: Hyperparameter = Fixed(100),
                 num_epochs: Hyperparameter = Fixed(10),
                 learning_rate: Hyperparameter = Fixed(0.001),
                 output_size: Hyperparameter = Fixed(10),
                 stop_early: Hyperparameter = Fixed(False),
                 patience: Hyperparameter = Fixed(5),
                 min_delta: Hyperparameter = Fixed(0.001),
                 width: Hyperparameter = Fixed(128),
                 depth: Hyperparameter = Fixed(10),
                 input_width: Hyperparameter = Fixed(28),
                 input_height: Hyperparameter = Fixed(28),
                 input_channels: Hyperparameter = Fixed(1),
                 kernel_size: Hyperparameter = Fixed(3),
                 stride: Hyperparameter = Fixed(1),
                 padding: Hyperparameter = Fixed(1),
                 pool: Hyperparameter = Fixed(False),
                 performance_measure: str = "weighted_f1"):
        """
        Instantiate the model adapter and define the hyperparameters for
        optimisation as instance variables.

        :param train_dataset: The dataset used for training
        :param eval_dataset: The dataset used for evaluation
        :param val_dataset: The dataset used for valuation
        :param batch_size: The batch size to use for training and evaluation
        :param num_epochs: The number of epochs to train on
        :param learning_rate: The learning rate for the Adam optimiser
        :param output_size: The dimension of the output layer
        :param stop_early: Whether or not to use early stopping
        :param patience: The number of epochs to wait for the loss delta to
                         improve
        :param min_delta: The minimum validation loss delta required to
                          continue training
        :param width: The number of filters per convolutional layer
        :param depth: The number of convolutional layers
        :param input_width: The width of the input in pixels
        :param input_height: The height of the input in pixels
        :param input_channels: The number of input colour channels
        :param kernel_size: The size of the filters (kernel_size x kernel_size)
        :param stride: The stride of the filters
        :param padding: The conv. padding
        :param pool: Whether or not to use max pooling after each conv. layer
        :param performance_measure: The performance measure to use in evaluation
        """
        super().__init__(train_dataset, eval_dataset, val_dataset, batch_size,
                         num_epochs, learning_rate, output_size, stop_early,
                         patience, min_delta, performance_measure)
        self.width, self.depth = width, depth
        self.input_width, self.input_height = input_height, input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride, self.padding, self.pool = stride, padding, pool

    def define(self):
        """Construct the model using the (potentially updated)
        hyperparameters."""
        self.model = CNN(self.width.value, self.depth.value,
                         self.input_width.value, self.input_height.value,
                         self.input_channels.value, self.kernel_size.value,
                         self.stride.value, self.padding.value,
                         self.output_size.value, self.pool.value
                         ).to(self.device)
