# ECOpt

Energy Consumption Optimiser (ECOpt) uses Bayesian optimisation to simultaneously tune model hyperparameters for energy-efficiency and accuracy. It quantifies the trade-off between energy-efficiency and accuracy in a Pareto frontier, enabling machine learning practitioners to minimise their environmental impact while meeting performance targets.

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

To optimise your model, extend the `model.Model` class to wrap it. Then use this class and your choice of `meter.Meter` to construct an `optimizer.Optimizer` and call it. Here is an example that defines some hyperparameters, optimises a (to be implemented) model and plots the Pareto frontier.

```python
from ecopt.model import Model
from ecopt.optimizer import Optimizer
from ecopt.hyperparameter import Range, Choice, Fixed
from ecopt.meter import CodeCarbonMeter


class MyModel(Model):
    """A wrapper for the model to optimise."""

    def __init__(self):
        """Define the hyperparameters for
        optimisation as instance variables."""
        self.hidden_size = Range(5, min=1, max=10)
        self.learning_rate = Range(0.001, min=0.0001, max=0.01, log_scale=True)
        self.depth = Choice(2, list(range(1, 6)), is_ordered=True)
        self.num_epochs = Fixed(3)

    def train(self):
        """Construct and train the model using the optimised and static
        hyperparameters."""
        # TODO: train model using hyperparameters
        return len(dataset) * self.num_epochs.value

    def evaluate(self) -> (float, int):
        """Evaluate the model, returning the utility (greater is better) and
        number of inferences."""
        # TODO: evaluate model
        return accuracy, len(dataset)


model = MyModel()
meter = CodeCarbonMeter()
optimizer = Optimizer(model, meter)
optimizer()
optimizer.plot_pareto_frontier()
```

Experiments are automatically tracked using [MLflow](https://mlflow.org) and saved in the working directory.
These can be viewed by running the following command and navigating to [localhost:5000](http://localhost:5000).
Experiment data can be exported in CSV format.

```bash
mlflow ui
```

To log your experiments to a remote MLflow server, specify the URI when instantiating a meter.
It is also possible to specify an experiment name, run name and run tags.

## Testing

The test suite can be evaluated with the following command.

```bash
python -m unittest
```

## Experiments

The experiments of the paper can be found in `experiments/`. First, install their dependencies.

```bash
pip install -r experiments/requirements.txt
```

The experiments can then be reproduced.
