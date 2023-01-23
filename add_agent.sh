#!/bin/bash -e

dataset=${1}  # "pubmed"
seed=${2}  # 1
sweep_id=${3}  # entity/project/id
n_agents=${4:-1}  # 1
gpu_name=${5:-"gypsum-1080ti"}  # "gypsum-1080ti"

for ((i = 1; i <= ${n_agents}; i++)); do
  JOB_DESC=${dataset}_sweep${seed}-${i} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=80G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --dataset="${dataset}" \
    --dataset_random_seed=${seed} \
    --wandb_sweep_params="wandb_configs/sweeps/e2e_main.json" \
    --skip_initial_eval --silent --wandb_sweep_id="${sweep_id}"
  echo "    Logs: jobs/${JOB_NAME}.err"
done
