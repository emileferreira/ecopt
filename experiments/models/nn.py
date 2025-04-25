import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import mlflow
from thop import profile

from ecopt.model import Model
from ecopt.hyperparameter import Fixed, Hyperparameter


class NeuralNetwork(torch.nn.Module):

    def __init__(self, input_size, hidden_size, depth, output_size):
        super(NeuralNetwork, self).__init__()
        assert depth >= 1
        self.layers = torch.nn.ModuleList()
        for layer in range(depth):
            is_first, is_last = layer == 0, layer == depth - 1
            layer_input_size = input_size if is_first else hidden_size
            layer_output_size = output_size if is_last else hidden_size
            self.layers.append(torch.nn.Linear(layer_input_size,
                                               layer_output_size))
            if not is_last:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralNetworkModel(Model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_dataset: Dataset, eval_dataset: Dataset,
                 hidden_size: Hyperparameter = Fixed(5),
                 learning_rate: Hyperparameter = Fixed(0.001),
                 depth: Hyperparameter = Fixed(2),
                 num_epochs: Hyperparameter = Fixed(3),
                 batch_size: Hyperparameter = Fixed(100),
                 input_size: Hyperparameter = Fixed(28 * 28),
                 output_size: Hyperparameter = Fixed(10)):
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.depth = depth
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

    def define(self):
        self.model = NeuralNetwork(self.input_size.value,
                                   self.hidden_size.value,
                                   self.depth.value,
                                   self.output_size.value).to(self.device)

    def train(self):
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
                    images = images.reshape(-1, self.input_size.value).to(
                        self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1 / len(dataloader))

    def evaluate(self) -> (float, int):
        image = self.eval_dataset[0][0]
        image = image.reshape(-1, self.input_size.value).to(
            self.device)
        macs, parameters = profile(self.model, inputs=(image,))
        mlflow.log_metrics({
            "parameters": parameters,
            "flops": 2 * macs
        })
        dataloader = DataLoader(dataset=self.eval_dataset,
                                batch_size=self.batch_size.value,
                                shuffle=False)
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(dataloader, unit="batch",
                                       desc="Evaluate"):
                images = images.reshape(-1, self.input_size.value).to(
                    self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                y_true.append(labels)
                y_pred.append(predictions)
        y_true, y_pred = np.concat(y_true), np.concat(y_pred)
        num_samples = len(self.eval_dataset)
        return f1_score(y_true, y_pred, average="weighted"), num_samples + 1
