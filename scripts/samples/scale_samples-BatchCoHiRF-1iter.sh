environment_name="cohirf"
experiment_python_location="/cohirf/cohirf/experiment/hpo_classification_clustering_experiment.py"

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate $environment_name

export TMPDIR="/tmp"
export OMP_NUM_THREADS="10"
export MKL_NUM_THREADS="10"
export OPENBLAS_NUM_THREADS="10"

string_array=(
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 100000 --seed_dataset 0 --hpo_seed 0"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 1000000 --seed_dataset 0 --hpo_seed 0"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 100000 --seed_dataset 1 --hpo_seed 1"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 1000000 --seed_dataset 1 --hpo_seed 1"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 100000 --seed_dataset 2 --hpo_seed 2"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 1000000 --seed_dataset 2 --hpo_seed 2"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 100000 --seed_dataset 3 --hpo_seed 3"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 1000000 --seed_dataset 3 --hpo_seed 3"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 100000 --seed_dataset 4 --hpo_seed 4"
"--max_threads 10 --model BatchCoHiRF-1iter --experiment_name scale_samples-BatchCoHiRF-1iter --n_jobs 10 --log_dir /home/users/belucci/cohirf/results/v2/logs --work_root_dir ${TMPDIR} --mlflow_tracking_uri http://clust9.ceremade.dauphine.lan:5002 --hpo_framework optuna --n_trials 20 --sampler tpe --pruner none --direction maximize --hpo_metric adjusted_rand --n_random 10000 --n_classes 5 --n_informative 10 --class_sep 9.486832980505138 --n_samples 1000000 --seed_dataset 4 --hpo_seed 4"
)
for string in "${string_array[@]}"; do
	echo "Running python ${experiment_python_location} $string"
	srun --exclusive -n 1 -c $SLURM_CPUS_PER_TASK python ${experiment_python_location} $string &
    sleep 2
done
wait

# clean up
rm -rf $TMPDIR
