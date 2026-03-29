#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=test
#SBATCH --job-name=marta_classifier_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH -o logs/marta_%j.out

echo "========================================"
echo "MARTA training started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

module purge
module load Miniforge3

# Go to repository root
cd /scratch/jsanchoz/MARTA || exit 1 # change this path

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/jsanchoz/.conda/envs/marta || exit 1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: load W&B secret if present
if [ -f /scratch/jsanchoz/MARTA/scripts/secrets/wandb.env ]; then
    echo "Loading optional W&B environment variables"
    source /scratch/jsanchoz/MARTA/scripts/secrets/wandb.env # change this path
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

echo "Training config: configs/train_classifier.yaml"
echo "Expert mode setting:"
grep -E "^expert_mode:" configs/train_classifier.yaml || true

python3.10 -m src.train.train_classifier \
    --config /scratch/jsanchoz/MARTA/configs/train_classifier.yaml

echo "========================================"
echo "MARTA training finished"
echo "Date: $(date)"
echo "========================================"