import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
                 output_size: Hyperparameter = Fixed(10)):
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_size = output_size

    def train(self):
        self.model = LeNet5(num_classes=self.output_size.value).to(self.device)
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=self.batch_size.value,
                                shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate.value)
        with tqdm(total=self.num_epochs.value, unit="epoch",
                  desc="Train") as progress:
            for epoch in range(self.num_epochs.value):
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1 / len(dataloader))

    def evaluate(self) -> (float, int):
        dataloader = DataLoader(dataset=self.eval_dataset,
                                batch_size=self.batch_size.value,
                                shuffle=False)
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(dataloader, unit="batch",
                                       desc="Evaluate"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                y_true.append(labels)
                y_pred.append(predictions)
        y_true, y_pred = np.concat(y_true), np.concat(y_pred)
        num_samples = len(self.eval_dataset)
        return f1_score(y_true, y_pred, average="weighted"), num_samples
