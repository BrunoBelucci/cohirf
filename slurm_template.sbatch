#!/bin/bash
#SBATCH -A gen15860@rome
#SBATCH -p rome
#SBATCH --licenses=fs_unshare,fs_work,fs_scratch
#SBATCH --job-name=classif_huge
#SBATCH --qos=normal
#SBATCH -c 2
#SBATCH --output=/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_scratch/recursive_clustering/results/classif_huge/sbatch_output/%x-%A_%a.out
#SBATCH --error=/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_scratch/recursive_clustering/results/classif_huge/sbatch_error/%x-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --array=0-14:1%15

# we are limited at 300 submitted jobs, so we will submit small array jobs and the last array will resubmit the next array job
array_start=0
array_end=639
array_step=1
array_simultaneous=15
array_batch=15
script_path="/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_work/recursive_clustering/run_scripts/classif_huge/classif_huge.sbatch"


# 1 model, 64 datasets, 10 fold -> 640 models, but slurm is configured to accept at maximum 1001 (array from 0-1000)
# then copy another template or modify this one accordingly

experiment_root_dir="/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_scratch/recursive_clustering/results/classif_huge"
experiment_work_root_dir="/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_scratch/recursive_clustering/results/classif_huge/${SLURM_JOB_NAME}"
experiment_save_root_dir="/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_work/recursive_clustering/results/classif_huge/${SLURM_JOB_NAME}"

# For job step/array
srun_output="${experiment_root_dir}/sbatch_output/srun-%x-%A_%a.out"
srun_error="${experiment_root_dir}/sbatch_error/srun-%x-%A_%a.err"
dbs_dir="${experiment_work_root_dir}/dbs"
mkdir -p $dbs_dir

# then copy another template or modify this one accordingly
environment_name="cohirf"
experiment_python_location="/ccc/cont003/home/dauphine/beluccib/gen15860_beluccib_work/recursive_clustering/recursive_clustering/experiment/hpo_classification_clustering_experiment.py"

# Create a dictionary with argument names and values
declare -A args_dict=(
# base
["models_nickname"]=""
["seeds_models"]=""
["experiment_name"]="${SLURM_JOB_NAME}"
["n_jobs"]=""
["models_params"]=""
["fits_params"]=""
["error_score"]=""
["timeout_fit"]=""
["timeout_combination"]=""
["log_dir"]="${experiment_work_root_dir}/logs"
#["log_file_name"]=""
["work_root_dir"]="${experiment_work_root_dir}/work"
["save_root_dir"]="${experiment_save_root_dir}/save"
#["mlflow_tracking_uri"]=""
["dask_cluster_type"]=""
["n_workers"]=""
["n_cores_per_worker"]=""
["n_processes_per_worker"]=""
["n_threads_per_worker"]=""
["n_cores_per_task"]=""
["n_processes_per_task"]=""
["n_threads_per_task"]=""
["dask_memory"]=""
["dask_job_extra_directives"]=""
["dask_address"]=""
["n_gpus_per_worker"]=""
["n_gpus_per_task"]=""
# hpo
["hpo_framework"]=""
["n_trials"]=""
["timeout_hpo"]=""
["timeout_trial"]=""
["max_concurrent_trials"]=""
["sampler"]=""
["pruner"]=""
["direction"]=""
["hpo_metric"]=""
# classification clustering
["n_samples"]=""
["n_random"]=""
["n_informative"]=""
["n_redundant"]=""
["n_repeated"]=""
["n_classes"]=""
["n_clusters_per_class"]=""
["weights"]=""
["flip_y"]=""
["class_sep"]=""
["hypercube"]=""
["shift"]=""
["scale"]=""
["shuffle"]=""
["seeds_dataset"]=""
["seeds_unified"]=""
["n_features"]=""
["pct_random"]=""
["add_outlier"]=""
)

declare -A bool_args_dict=(
# base
["create_validation_set"]=0
["do_not_clean_work_dir"]=0
["do_not_log_to_mlflow"]=0
["do_not_check_if_exists"]=0
["do_not_retry_on_oom"]=0
["raise_on_fit_error"]=0
)

declare -A array_args_dict=(
# Note that bash does not allow arrays inside dictionaries, so we will use strings with ',' as separators and '-' when
# we want multiple values for the same argument (basically they will be replaced by ' ')
# base

# in this example we are expecting the combinations Model1+0-1, Model1+2-3, Model1+4, Model2+0-1, Model2+2-3, Model2+4
)

# bash does not necessarily keep the order of the keys in the dictionary, so we will specify the order here
declare -a array_args_dict_order=()

# Construct the argument string
args_str=""
for key in "${!args_dict[@]}"; do
  if [ -n "${args_dict[$key]}" ]; then
    args_str="$args_str --$key ${args_dict[$key]}"
  fi
done

# Add arguments strings that are boolean
for key in "${!bool_args_dict[@]}"; do
  if [ "${bool_args_dict[$key]}" -eq 1 ]; then
    args_str="${args_str} --${key}"
  fi
done

# Construct the cartesian product of the arrays
# the idea is to create a string like {Model1,Model2}+{0-1,2-3,4} and then evaluate it to get the cartesian product
# using bash's brace expansion
string_for_cartesian_product=""
for key in "${array_args_dict_order[@]}"; do
  str_array=${array_args_dict[$key]}
  n_elements=$(echo $str_array | tr ',' ' ' | wc -w)
  if [ ${n_elements} -eq 0 ]; then
    continue
  elif [ ${n_elements} -eq 1 ]; then
    string_for_cartesian_product="${string_for_cartesian_product}+${str_array}"
  else
    string_for_cartesian_product="${string_for_cartesian_product}+{${str_array}}"
  fi
done

# Remove the first '+' character
string_for_cartesian_product=${string_for_cartesian_product:1}

# Evaluate the string to get the cartesian product
cartesian_product=$(eval echo $string_for_cartesian_product)

# Split the string into an array (1 combination per element)
IFS=' ' read -r -a cartesian_product <<< "${cartesian_product}"
# cartesian_product is now an array like ["Model1+0-1", "Model1+2-3", "Model1+4", "Model2+0-1", "Model2+2-3", "Model2+4"]

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate $environment_name

# alternatively we could also run an array job and execute the combination given by the SLURM_ARRAY_TASK_ID
i_combination=$SLURM_ARRAY_TASK_ID
if [ "${i_combination}" -ge "${#cartesian_product[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID is greater than the number of combinations"
  exit 1
fi
string_combination=""
IFS='+' read -r -a combination <<< "${cartesian_product[$i_combination]}"
i_arg_name=0
for key in "${array_args_dict_order[@]}"; do
  value=${combination[$i_arg_name]//[-]/ }  # replace '-' by ' '
  string_combination="${string_combination} --${key} ${value}"
  i_arg_name=$((i_arg_name+1))
done
mlflow_tracking_uri="sqlite:////${dbs_dir}/${cartesian_product[$i_combination]}.db"
log_file_name="${SLURM_JOB_NAME}-${cartesian_product[$i_combination]}"
srun --exclusive -n 1 -c ${SLURM_CPUS_PER_TASK} --output=${srun_output} --error=${srun_error} python ${experiment_python_location} ${args_str} ${string_combination} --mlflow_tracking_uri ${mlflow_tracking_uri} --log_file_name ${log_file_name}
wait

# If we are at the last array task, submit the next array job
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
  next_array_start=$(($SLURM_ARRAY_TASK_MIN+array_batch))
  # next_array_end will be the minimum between array_end and next_array_start+array_batch
  next_array_end=$(( array_end < next_array_start+array_batch ? array_end : next_array_start+array_batch ))
  # if next_array_start <= array_end, then we submit job, otherwise we are done
  if [ "${next_array_start}" -le "${next_array_end}" ]; then
    sbatch --array=${next_array_start}-${next_array_end}:${array_step}%${array_simultaneous} ${script_path}
  fi
fi