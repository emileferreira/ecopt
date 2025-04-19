from ecopt.meter import CodeCarbonMeter
from ecopt.hyperparameter import Fixed
from models.cnn import LeNet5Model
from datasets.mnist import MNIST


mlflow_tracking_uri = None
run_tags = {
    "machine": "laptop",
    "rapl": True
}
dataset = MNIST()
meter = CodeCarbonMeter(
    experiment_name="RAPL-constant",
    mlflow_tracking_uri=mlflow_tracking_uri)
for _ in range(35):
    model = LeNet5Model(train_dataset=dataset.train_dataset,
                        eval_dataset=dataset.eval_dataset)
    meter(model, utility_measure="weighted_f1", run_tags=run_tags)
meter = CodeCarbonMeter(
    experiment_name="RAPL-batch",
    mlflow_tracking_uri=mlflow_tracking_uri)
for exp in range(4, 14):
    batch_size = Fixed(2**exp)
    model = LeNet5Model(train_dataset=dataset.train_dataset,
                        eval_dataset=dataset.eval_dataset,
                        batch_size=batch_size)
    meter(model, utility_measure="weighted_f1", run_tags=run_tags)
meter = CodeCarbonMeter(
    experiment_name="RAPL-epoch",
    mlflow_tracking_uri=mlflow_tracking_uri)
for num_epochs in range(1, 11):
    model = LeNet5Model(train_dataset=dataset.train_dataset,
                        eval_dataset=dataset.eval_dataset,
                        num_epochs=Fixed(num_epochs))
    meter(model, utility_measure="weighted_f1", run_tags=run_tags)
