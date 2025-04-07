import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from ecopt.model import Model
from ecopt.hyperparameter import Range, Fixed
from ecopt.meter import CodeCarbonMeter


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

    batch_size = 100
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = 10

    def __init__(self):
        # define hyperparameters
        self.learning_rate = Range(0.001, min=0.0001, max=0.01, log_scale=True)
        self.num_epochs = Fixed(1)
        self.model = LeNet5(num_classes=self.output_size).to(self.device)
        # define train dataset
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True,
            transform=torchvision.transforms.ToTensor())
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        # calculate weights for cross-entropy loss
        self.cle_weight = torch.zeros(self.output_size)
        for _, labels in self.train_dataloader:
            self.cle_weight += torch.bincount(labels)
        # define evaluation dataset
        self.eval_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False,
            transform=torchvision.transforms.ToTensor())
        self.eval_dataloader = torch.utils.data.DataLoader(
            dataset=self.eval_dataset, batch_size=self.batch_size,
            shuffle=False)

    def train(self):
        criterion = torch.nn.CrossEntropyLoss(self.cle_weight)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate.value)
        with tqdm(total=self.num_epochs.value, unit="epoch",
                  desc="Train") as progress:
            for epoch in range(self.num_epochs.value):
                for images, labels in self.train_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1 / len(self.train_dataloader))

    def evaluate(self) -> (float, int):
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.eval_dataloader, unit="batch",
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


model = LeNet5Model()
experiment_name = "RAPL"
mlflow_tracking_uri = None
run_tags = {
    "machine": "laptop",
    "rapl": False
}
meter = CodeCarbonMeter(
    experiment_name=experiment_name,
    mlflow_tracking_uri=mlflow_tracking_uri,
    log_level="ERROR")
print(meter(model, utility_measure="weighted_f1", run_tags=run_tags))
