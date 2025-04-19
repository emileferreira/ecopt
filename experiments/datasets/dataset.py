DATA_DIR = "./data"


class DatasetWrapper:

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def eval_dataset(self):
        raise NotImplementedError
