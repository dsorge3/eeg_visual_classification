#!/bin/bash
#SBATCH --job-name=DcGan
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

DATASET="EEG_occhi"
MODEL_NAME="DcGan"
MODE="train"

###### TRAIN
# Use --mode train
# Comment --path
# Comment --weights

###### CONTINUE TRAINING
# Add --load_path
# Add --weights
# Add --resume_epoch epoch_number

###### VISUALIZATION
# Add MODEL_PATH to load model and use visualization
# Use --mode vis
# Add --path
# Add --weights

##### RESUME
#--log_dir /home/d.sorge/eeg_visual_classification/dcgan/logs/fit/DcGan_EEG/20220516-120840 \
#--load_path /home/d.sorge/eeg_visual_classification/dcgan/logs/fit/DcGan_EEG/20220516-120840/checkpoint_234.pth \

#--lstm_path /home/d.sorge/eeg_visual_classification/eeg_visual_classification_main/lstm_4class_128_subject0_epoch_80.pth \

python -u dcgan.py \
--name "${MODEL_NAME}_${DATASET}" \
--mode ${MODE} \
--model "${MODEL_NAME}_${DATASET}" \
--batch_size 8 \
--dataset ${DATASET} \
--lstm_path /home/d.sorge/eeg_visual_classification/eeg_visual_classification_main/lstm_eegOcchi_128_subject0_epoch_21.pth \
--num_epochs 1200
