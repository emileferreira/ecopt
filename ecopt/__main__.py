import torch
import torchvision
from tqdm import tqdm

from .model import Model
from .optimizer import Optimizer
from .hyperparam import Hyperparam


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


class MyModel(Model):

    batch_size = 100
    data_dir = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 28 * 28
    output_size = 10
    num_epochs = 3

    def __init__(self):
        self.hidden_size = Hyperparam(500)
        self.depth = Hyperparam(2, 0, 10)
        self.learning_rate = Hyperparam(0.001)

    def train(self):
        self.model = NeuralNetwork(self.input_size,
                                   self.hidden_size.value,
                                   self.depth.value,
                                   self.output_size).to(self.device)
        dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True,
            transform=torchvision.transforms.ToTensor())
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate.value)
        with tqdm(total=self.num_epochs, unit="epoch",
                  desc="Train") as progress:
            for epoch in range(self.num_epochs):
                for images, labels in dataloader:
                    images = images.reshape(-1, self.input_size).to(
                        self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1 / len(dataloader))
        return len(dataset) * self.num_epochs

    def evaluate(self) -> (float, int):
        dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False,
            transform=torchvision.transforms.ToTensor())
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False)
        num_correct = 0
        num_samples = len(dataset)
        with torch.no_grad():
            for images, labels in tqdm(dataloader, unit="batch",
                                       desc="Evaluate"):
                images = images.reshape(-1, self.input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                num_correct += (predictions == labels).sum().item()
        accuracy = num_correct / num_samples
        return accuracy, num_samples


model = MyModel()
optimizer = Optimizer(model, log_level="ERROR")
optimizer(1)
