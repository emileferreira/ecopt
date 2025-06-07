# ECOpt

Energy Consumption Optimiser (ECOpt) is a hyperparameter tuner that optimises for energy efficiency and utility.
ECOpt quantifies the compromise between these metrics as an interpretable Pareto frontier.
This enables ML practitioners to make informed decisions about energy cost and environmental impact, while maximising the benefit of their models and complying with new regulations.

## Requirements

- Python 3.10+

## Setup

Optionally, create and enter a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the ECOpt package. Omit the `-e` flag if you do not want changes in this directory to be reflected by the installed package. If the installation fails due to insufficient space, specify the temporary directory: `export TMPDIR="/var/tmp"`.

```bash
pip install -v -e .
```

## Usage

Extend the `model.Model` class to wrap your model to be optimised and define the hyperparameters as instance variables. This prepares the model for use with the ECOpt meter, and the model hyperparameters for the ECOpt optimiser. This example wraps a `NeuralNetwork` model. The implementation of the model, training loop and evaluation function is omitted for brevity.

```python
from ecopt.model import Model
from ecopt.hyperparameter import Range, Choice, Fixed
from ecopt.meter import CodeCarbonMeter
from ecopt.optimizer import Optimizer


class NeuralNetwork:
    """A neural network model."""

    __init__(self, hidden_size: int, depth: int):
        """Construct a model of the specified size."""
        ...


class NeuralNetworkModel(Model):
    """An adapter of `NeuralNetwork` for ECOpt."""

    def __init__(self):
        """Instantiate a model adapter and define the hyperparameters for
        optimisation as instance variables."""
        self.hidden_size = Range(28, min=28, max=28*28)
        self.learning_rate = Range(0.001, min=0.0001, max=0.01, log_scale=True)
        self.depth = Choice(2, list(range(1, 6)), is_ordered=True)
        self.num_epochs = Fixed(10)

    def define(self):
        """Construct the model using the hyperparameters."""
        self.model = NeuralNetwork(self.hidden_size.value, self.depth.value)

    def train(self):
        """Train the model."""
        for epoch in range(self.num_epochs.value):
            ...

    def evaluate(self) -> (float, int):
        """Evaluate the model, returning the utility and number of samples."""
        ...
        return f1, len(dataset)
```

You can optionally use a `meter.Meter` to measure the energy efficiency and utility of your model without optimisation.
The runs are optionally tagged to help identify them later. The country code is used to estimate the carbon emissions of the energy used.

```Python
run_tags = {"machine": "laptop", "rapl": True}
model = NeuralNetworkModel()
meter = CodeCarbonMeter(experiment_name="Metrics-depth", country_iso_code="GBR")
meter(model, utility_measure="weighted_f1", run_tags=run_tags)
```

Then use your wrapped model and meter to construct an `optimizer.Optimizer`, and invoke it. In this example, the optional objective metrics and a threshold for the utility are specified.

```Python
optimizer = Optimizer(model, meter, utility_measure="weighted_f1",
                      efficiency_measure="samples_per_j")
optimizer(num_init_steps=5, num_opt_steps=20,
          utility_threshold=0.75, run_tags=run_tags)
```

You can then plot an interactive Pareto frontier. This will open in a web browser.

```Python
optimizer.plot_pareto_frontier()
```

## Experiment tracking

Experiments are automatically tracked using [MLflow](https://mlflow.org) and saved in the working directory.
These can be viewed by running the following command and navigating to [localhost:5000](http://localhost:5000).

```bash
mlflow ui
```

To log your experiments to a remote MLflow server, specify the URI when instantiating the meter.
It is also possible to specify an experiment name, run name and run tags.
Experiment data can be exported in CSV format.

## Testing

The suite of unit tests can be evaluated with the following command.

```bash
python -m unittest
```

## Experiments

The experiments of the project can be found in `experiments/`. First, install their dependencies.

```bash
pip install -r experiments/requirements.txt
```

The experiments can then be reproduced by running their scripts.
