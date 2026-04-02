#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=marta_inference_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH -o logs/marta_infer_%j.out

echo "========================================"
echo "MARTA batch inference started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

module purge
module load Miniforge3

# Go to repository root
cd /scratch/jsanchoz/MARTA || exit 1

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marta || exit 1

echo "Python: $(which python)"
python --version
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# python -m src.inference.batch_grid \
#     --folder_images /scratch/jsanchoz/MARTA/data/images \
#     --ckpt /scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE/best_stage3_full.pth \
#     --outdir /scratch/jsanchoz/MARTA/inference_batch_out \
#     --config /scratch/jsanchoz/MARTA/configs/inference.yaml

python -m src.inference.single_image \
    --image /scratch/jsanchoz/MARTA/data/images/IM1313.png \
    --ckpt /scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE_10_epochs/mlp_128_d0.5_stack_dual_20260319_205828/best_stage3_full.pth \
    --outdir /scratch/jsanchoz/MARTA/experiments/MARTA_MULTIINPUT_SINGLE_10_epochs/infer_test_slurm \
    --config /scratch/jsanchoz/MARTA/configs/inference.yaml

echo "========================================"
echo "MARTA batch inference finished"
echo "Date: $(date)"
echo "========================================"