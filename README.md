# Classical Out-of-Distribution Detection Methods Benchmark in Text Classification Tasks

This repository provides a benchmark for classical out-of-distribution (OOD) detection methods in text classification tasks. It evaluates the performance of various OOD detection techniques on text classification datasets.

## Installation

To set up the environment, follow these steps:

1. Create a Python 3.10+ virtual environment.
2. Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## DVC

Data for the benchmark can be downloaded using `dvc`. To download the data, execute the following command:

```bash
dvc pull
```

## Configuration

To modify the experimental configurations, you can edit the files in the `[params/](params)` directory.

## Running Experiments

To train the models, use the following command:

```bash
PYTHONPATH=. python scripts/train.py
```

To evaluate the models, use the following command:

```bash
PYTHONPATH=. python scripts/evaluate.py
```

## Results

The results of the experiments will be stored in the `[experiments/](experiments)` directory. Additionally, you have the option to store the results on the WandB cloud account for further analysis and visualization.