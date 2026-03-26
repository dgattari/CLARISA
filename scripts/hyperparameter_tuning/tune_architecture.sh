#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=test
#SBATCH --job-name=marta_tune_arch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH -o logs/marta_tune_arch_%j.out

echo "========================================"
echo "MARTA architecture tuning started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

module purge
module load Miniforge3

cd /scratch/jsanchoz/MARTA || exit 1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/jsanchoz/.conda/envs/marta || exit 1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -f /scratch/jsanchoz/MARTA/scripts/secrets/wandb.env ]; then
    echo "Loading optional W&B environment variables"
    source /scratch/jsanchoz/MARTA/scripts/secrets/wandb.env
fi

cleanup_wandb() {
    echo "Cleaning W&B cache directories..."
    rm -rf /scratch/jsanchoz/MARTA/.wandb_cache/*
    rm -rf /scratch/jsanchoz/MARTA/.wandb_data/*
    rm -rf /scratch/jsanchoz/MARTA/.wandb_artifacts/*
}

trap cleanup_wandb EXIT

echo "Python: $(which python)"
python --version
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

python3.10 -c "import src; print('src package found')"
python3.10 -c "import albumentations; print('albumentations ok')"
python3.10 -c "import optuna; print('optuna ok')"

echo "Tuning config: /scratch/jsanchoz/MARTA/configs/hyperparameter_tuning/tune_architecture.yaml"

python3.10 -m src.train.tune_classifier \
    --config /scratch/jsanchoz/MARTA/configs/hyperparameter_tuning/tune_architecture.yaml

echo "========================================"
echo "MARTA architecture tuning finished"
echo "Date: $(date)"
echo "========================================"