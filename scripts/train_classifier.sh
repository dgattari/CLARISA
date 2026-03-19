#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
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
cd /scratch/jsanchoz/MARTA || exit 1 # change this

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marta || exit 1

echo "Python: $(which python)"
python --version
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -c "import src; print('src package found')"
python -m src.train.train_classifier \
    --config configs/train_classifier.yaml

echo "========================================"
echo "MARTA training finished"
echo "Date: $(date)"
echo "========================================"