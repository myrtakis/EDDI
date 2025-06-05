# EDDI
EDDI: Explainable Drift Detection using Influence

## Installation Instructions

### Setting up the virtual environment and packages

Create a virtual environment using conda with Python 3.10.
```
conda create -n eddi python=3.10
conda activate eddi
```
Install [poetry](https://python-poetry.org/docs/) for dependency management and conflict resolution. 
To download poetry into a specific folder run
```
curl -sSL https://install.python-poetry.org | POETRY_HOME=/hri/storage/user/<USER>/etc/my_poetry python3 -
```

To install the necessary Python libs, go to the project's main directory, i.e., where the *.toml* file is on the same dir level and run
```
poetry install
```

Download the datasets 
```
python download_ts_classification.py 
```

