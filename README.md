# Remote Sensing Fusion

## 1. Prerequisites

To run this project locally, you will need:

* Conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/))

## 2. Binder Setup Instructions

To run this project in a web browser, click the icon below to launch the project with Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/calekochenour/remote-sensing-fusion/master)

Binder will open a Jupyter Notebook in the current web browser.

## 3. Local Setup Instructions

To run this analysis from a terminal, navigate to the folder containing the local repository.

Local instructions assume the user has cloned or forked the GitHub repository.

### Create and Activate Conda Environment

From the terminal, you can create and activate the project Conda environment.

Create environment:

```bash
conda env create -f environment.yml
```

Activate environment:

```bash
conda activate remote-sensing-fusion
```

### Open Jupyter Notebook

From the terminal, you can run the analysis and produce the project outputs.

Open Jupyter Notebook:

```bash
jupyter notebook
```

## 4. Contents

The project contains folders for all stages of the workflow as well as other files necessary to run the analysis.

### `01-code-scripts/`

Contains all Python scripts and Jupyter Notebooks required to run the analysis.

### `02-raw-data/`

Contains all original/unprocessed data.

### `03-processed-data/`

Contains all processed/created data.

### `04-graphics-outputs/`

Contains all figures.

### `05-papers-writings/`

Contains all paper/report files.

### `environment.yml`

Contains the information required to create the Conda environment.
