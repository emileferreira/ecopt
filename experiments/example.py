from ecopt.optimizer import Optimizer
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Range
from models.neural_network import NeuralNetworkModel
from datasets.mnist import MNIST


dataset = MNIST()
hidden_size = Range(5, min=1, max=10)
depth = Range(2, min=1, max=10)
model = NeuralNetworkModel(train_dataset=dataset.train_dataset,
                           eval_dataset=dataset.eval_dataset,
                           hidden_size=hidden_size,
                           depth=depth)
meter = CodeCarbonMeter(log_level="ERROR")
optimizer = Optimizer(model, meter, utility_measure="weighted_f1")
optimizer(num_init_steps=4, num_opt_steps=10, utility_threshold=0,
          efficiency_threshold=0)
optimizer.plot_pareto_frontier()
