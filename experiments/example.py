from ecopt.optimizer import Optimizer
from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Choice
from models.vit import ViTMNISTModel


model = ViTMNISTModel(batch_size=Choice(64, [64, 128]))
meter = CodeCarbonMeter(log_level="ERROR")
optimizer = Optimizer(model, meter, utility_measure="weighted_f1")
optimizer(num_init_steps=4, num_opt_steps=10, utility_threshold=0,
          efficiency_threshold=0)
optimizer.plot_pareto_frontier()
