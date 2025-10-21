#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=JEPA
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/GRAM-M
HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-mamba-trainer


python3 train.py data=audioset data.clean_data_ratio=0.0