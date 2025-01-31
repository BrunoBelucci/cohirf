# CoHiRF : A Scalable and Interpretable Clustering Framework for High-Dimensional Data

This is the supplementary code for the paper "CoHiRF: A Scalable and Interpretable Clustering Framework for 
High-Dimensional Data"


## Installing the requirements

To install the requirements, run the following command:

```bash
conda env create -f environment.yml
```

We have also a supplementary dependency on an internal library `ml_experiments`. To install it go to its directory and 
run:

```bash
pip install -e .
```

## Running the experiments

We have provided the commands to run all the experiments in the bash scripts 'run_*.sh', where * is the name of the 
experiment. Note that as mentioned in the 
paper the code can take a long time to run and depending on the machine setup will not run for some combinations of 
algorithms and datasets. Even though we provide the code to run everything at once for each experiment, we recommend
running each combination of model and dataset separately.

## Repository Structure

The repository is structured as follows:

- cohirf: package containing the code for the CoHiRF model and experiments.
  - 'experiment': Contains the code to run each experiment in a separate file. The experiment code heavily relies on 
the `ml_experiments` library.
  - 'models': Contains the code for every model used in the experiments.
    - 'cohirf': Contains the code for the CoHiRF model.

- results: supplementary notebooks to visualize the results of the experiments.
