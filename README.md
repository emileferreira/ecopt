# ECOpt

Energy Consumption Optimiser (ECOpt) uses Bayesian optimisation to simultaneously tune model hyperparameters for energy-efficiency and accuracy. It quantifies the trade-off between energy-efficiency and accuracy in a Pareto frontier, enabling machine learning practitioners to minimise their environmental impact while meeting performance targets.

## Requirements

- Python 3.10 or later

## Setup

Create a Python virtual environment and install the requirements.

```Bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

An example model can be optimised using the following command.

```Bash
python -m ecopt
```

To optimise your own model, extend the `model.Model` class to wrap your model. Then use it to construct an `optimizer.Optimizer`.
