# CoHiRF : A Scalable and Interpretable Clustering Framework for High-Dimensional Data

This is the main code for CoHiRF, which is in current development. 


## Installing the requirements

We recommend using a virtual environment created with conda from the project package-list.txt with the following command (this way we ensure that we have for example the correct version of python):

```bash
conda create -n cohirf --file package-list.txt
```

Then, activate the environment:

```bash
conda activate cohirf
```

and install the project with pip:

```bash
pip install -e .
```

We have also a supplementary dependency on an internal library `ml_experiments`. To install, download it from
https://github.com/BrunoBelucci/ml_experiments, go to its directory and run:

```bash
pip install -e .
```

## Repository Structure

The repository is structured as follows:

- cohirf: package containing the code for the CoHiRF model and experiments.
  - 'experiment': Contains the code to run each experiment in a separate file. The experiment code heavily relies on 
the `ml_experiments` library.
  - 'models': Contains the code for every model used in the experiments.
    - 'cohirf': Contains the code for the CoHiRF model.
  - 'metrics.py': Contains the code for some metrics implemented in dask for large datasets.

- analyze_results: some notebooks to analyze experiment results.
- development_notebooks: some notebooks to test the code and run small experiments.
- teste: unit tests for the code.