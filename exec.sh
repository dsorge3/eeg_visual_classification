#!/bin/bash
#SBATCH --job-name=LSTM4CLASS
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --output=slurm-%j-%x.out

module load anaconda/3
module load cuda/10.1

source /home/d.sorge/.bashrc
conda activate venv

DATASET="EEG_4CLASS"
MODEL_NAME="LSTM"

#--log_dir /home/d.sorge/eeg_visual_classification/eeg_visual_classification-main/logs/fit/LSTM_EEG/20220214-193039/

python eeg_signal_classification.py \
--name "${MODEL_NAME}_${DATASET}" \
--epochs 200
