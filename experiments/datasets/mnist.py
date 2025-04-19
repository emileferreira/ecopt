import torchvision
import torch

from .dataset import DATA_DIR, DatasetWrapper


class MNIST(DatasetWrapper):
    """A wrapper for the MNIST dataset."""

    def __init__(self):
        DATASET_KWARGS = {
            "root": DATA_DIR,
            "download": True,
            "transform": torchvision.transforms.ToTensor()
        }
        self.train = torchvision.datasets.MNIST(
            **DATASET_KWARGS, train=True)
        self.eval = torchvision.datasets.MNIST(
            **DATASET_KWARGS, train=False)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.train

    @property
    def eval_dataset(self) -> torch.utils.data.Dataset:
        return self.eval
