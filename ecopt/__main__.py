from .model import Model


class MyModel(Model):
    pass


m = MyModel({"x": 1, "y": 2})
print(m.hyperparams)
