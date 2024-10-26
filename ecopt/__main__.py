from .model import Model
from .optimizer import Optimizer


class MyModel(Model):
    pass


model = MyModel({"x": 1, "y": 2})
optimizer = Optimizer(model)
print(optimizer.model.hyperparams)
