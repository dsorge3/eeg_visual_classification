#!/bin/bash
#SBATCH --job-name=VisualImag
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=4
#SBATCH --output=slurm-%j-%x.out

module load anaconda/3
module load cuda/10.1

source /home/d.sorge/.bashrc
conda activate venv

python train_derived.py --dataset eegdataset --gen_model ViT_custom_rp --dis_model ViT_custom_scale2_rp_noise --max_iter 500000 --exp_name eeg_train --grow_steps 0 0
