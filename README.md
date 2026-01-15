# CoHiRF : A Scalable and Interpretable Clustering Framework for High-Dimensional Data

This is the main code for CoHiRF, which is in current development. 


## Installing the requirements

We recommend using a virtual environment created with conda from the project package-list.txt with the following command (this way we ensure that we have for example the correct version of python):

```bash
conda create -n cohirf --file package-list.txt -c conda-forge
```
conda create -n cocohirf --file package-list.txt -c conda-forge
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
    - 'batch_cohirf': Contains the code for the BatchCoHiRF model.
  - 'metrics.py': Contains the code for some metrics implemented in dask for large datasets.

- analyze_results: some notebooks to analyze experiment results.
- development_notebooks: some notebooks to test the code and run small experiments.
- test: unit tests for the code.

## Running the experiments

To run the experiments, you can use the scripts inside the `scripts` folder. Note that we provide the script to run the
full experiment in one go, but in practice it is better to run the experiments in smaller chunks and check if they have
run successfully, especially because some combinations of models and datasets can take a long time to run or need more
memory than available in the machine, which can lead to the process being killed by the system. Once the experiments are
run, you can analyze the results using the notebooks in the `analyze_results` folder, adapting the "Load mlflow runs"
section to the location of the results of your experiments.