# Automated Benchmark Analysis on Various Tabular Models

![Last Commit](https://img.shields.io/github/last-commit/sonnguyen129/Tabular-Experiments)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/sonnguyen129/Tabular-Experiments/graphs/commit-activity))
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

A collection of state-of-the-art models on tabular data. Model performance can be checked with few lines of scripting. List of supported models can be seen here.

## How to use

### Installation

Clone this repository and check the ```requirements.txt```:
```shell
git clone https://github.com/sonnguyen129/Tabular-Experiments
cd Tabular-Experiments
pip install -r requirements.txt
```

### Run a single model on a single dataset

To run a single model on a single dataset call:

``python train.py --config/<config-file of the dataset>.yml --model_name <Name of the Model>``

All parameters set in the config file, can be overwritten by command line arguments, for example:

- ``--optimize_hyperparameters`` Uses [Optuna](https://optuna.org/) to run a hyperparameter optimization. If not set, the parameters listed in the `best_params.yml` file are used.

- ``--n_trails <number trials>`` Number of trials to run for the hyperparameter search

- ``--epochs <number epochs>`` Max number of epochs

- ``--use_gpu`` If set, available GPUs are used (specified by `gpu_ids`)

- ... and so on. All possible parameters can be found in the config files or calling: 
``python train.y -h``

If you are using the docker container, first enter the right conda environment using `conda activate <env name>` to 
have all required packages. The `train.py` file is in the `opt/notebooks/` directory.

--------------------------------------

## Add new models

Every new model should inherit from the base class `BaseModel`. Implement the following methods:

- `def __init__(self, params, args)`: Define your model here.
- `def fit(self, X, y, X_val=None, y_val=None)`: Implement the training process. (Return the loss and validation history)
- `def predict(self, X)`: Save and return the predictions on the test data - the regression values or the concrete classes for classification tasks
- `def predict_proba(self, X)`: Only for classification tasks. Save and return the probability distribution over the classes.
- `def define_trial_parameters(cls, trial, args)`: Define the hyperparameters that should be optimized.
- (optional) `def save_model`: If you want to save your model in a specific manner, override this function to.

Add your `<model>.py` file to the `models` directory and do not forget to update the `models/__init__.py` file.

----------------------------------------------

## Add new datasets

Every dataset needs a config file specifying its features. Add the config file to the `config` directory.

Necessary information are:
- *dataset*: Name of the dataset
- *objective*: Binary, classification or regression task
- *direction*: Direction of optimization. In the current implementation the binary scorer returns the AUC-score,
hence, should be maximized. The classification scorer uses the log loss and the regression scorer mse, therefore
both should be minimized.
- *num_features*: Total number of features in the dataset
- *num_classes*: Number of classes in classification task. Set to 1 for binary or regression task.
- *cat_idx*: List the indices of the categorical features in your dataset (if there are any).

It is recommended to specify the remaining hyperparameters here as well.

----------------------------

## Available models
| Name | Paper |
| :--: | :--: |
| LinearModel | Linear / Logistic Regression |
| TabNet | [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) |
| TabTransformer | [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)

## Acknowledgements
Specially thanks to [Ms.@kathrinse](https://github.com/kathrinse) for [this codebase](https://github.com/kathrinse/TabSurvey). If you interested in her work, please cite:
```
@article{borisov2021deep,
  title={Deep neural networks and tabular data: A survey},
  author={Borisov, Vadim and Leemann, Tobias and Se{\ss}ler, Kathrin and Haug, Johannes and Pawelczyk, Martin and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2110.01889},
  year={2021}
}
```
