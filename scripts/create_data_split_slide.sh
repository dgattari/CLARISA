#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=test
#SBATCH --job-name=marta_split_slide
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --nodes=1
#SBATCH -o logs/marta_split_slide_%j.out

echo "========================================"
echo "MARTA slide split creation started"
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

echo "Python: $(which python)"
python --version

python3.10 -c "import src; print('src package found')"
python3.10 -c "import albumentations; print('albumentations ok')"

echo "Split config: /scratch/jsanchoz/MARTA/configs/data_split_slide_v1.yaml"

python3.10 -m src.train.create_data_split \
    --config /scratch/jsanchoz/MARTA/configs/data_split_slide_v1.yaml

echo "========================================"
echo "MARTA slide split creation finished"
echo "Date: $(date)"
echo "========================================"