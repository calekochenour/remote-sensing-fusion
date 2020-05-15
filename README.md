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

## 4. Run the Analysis

Follow these steps upon completion of the **Binder Setup Instructions** or **Local Setup Instructions** to run the analysis in Jupyter Notebook:

* Navigate to the `01-code-scripts` folder;

* Click on the `penn-state-black-marble-radiance.ipynb` file;

* Select the `Kernal` tab and then the `Restart & Run All` option from the top of the browser page; and,

* Select the `Restart and Run All Cells` button in the pop-up window.

Once the user selects the `Restart and Run All Cells` button, the workflow will run all code, export figures, and display the results of the analysis.

## 5. Demos

### Run Analysis

![Run Analysis Demo](06-workflow-demos/penn-state-black-marble-radiance-demo-run-analysis.gif)

### View Results

![View Results Demo](06-workflow-demos/penn-state-black-marble-radiance-demo-view-results.gif)

## 6. Contents

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

### `06-workflow-demos/`

Contains all files for workflow demonstrations.

### `environment.yml`

Contains the information required to create the Conda environment.
