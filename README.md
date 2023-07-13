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

## Citation

```text
@inproceedings{baran-etal-2023-classical,
    title = "Classical Out-of-Distribution Detection Methods Benchmark in Text Classification Tasks",
    author = "Baran, Mateusz  and
      Baran, Joanna  and
      W{\'o}jcik, Mateusz  and
      Zi{\k{e}}ba, Maciej  and
      Gonczarek, Adam",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-srw.20",
    pages = "119--129",
    abstract = "State-of-the-art models can perform well in controlled environments, but they often struggle when presented with out-of-distribution (OOD) examples, making OOD detection a critical component of NLP systems. In this paper, we focus on highlighting the limitations of existing approaches to OOD detection in NLP. Specifically, we evaluated eight OOD detection methods that are easily integrable into existing NLP systems and require no additional OOD data or model modifications. One of our contributions is providing a well-structured research environment that allows for full reproducibility of the results. Additionally, our analysis shows that existing OOD detection methods for NLP tasks are not yet sufficiently sensitive to capture all samples characterized by various types of distributional shifts. Particularly challenging testing scenarios arise in cases of background shift and randomly shuffled word order within in domain texts. This highlights the need for future work to develop more effective OOD detection approaches for the NLP problems, and our work provides a well-defined foundation for further research in this area.",
}
```
