#!/bin/bash
#SBATCH --job-name=VisualImagCifarLstm
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --mem=44000
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j-%x.out

module load anaconda/3
module load cuda/10.1

source /home/d.sorge/.bashrc
conda activate venv

python train_derived.py \
-gen_bs 128 \
-dis_bs 64 \
--dist-url 'tcp://localhost:10641' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--dataset eegdataset \
--bottom_width 8 \
--img_size 32 \
--max_iter 500000 \
--gen_model ViT_custom_rp \
--dis_model ViT_custom_scale2_rp_noise \
--df_dim 384 \
--d_heads 4 \
--d_depth 3 \
--g_depth 5,4,2 \
--dropout 0 \
--latent_dim 256 \
--gf_dim 1024 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 8 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 4 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--diff_aug translation,cutout,color \
--load_path /home/d.sorge/eeg_visual_classification/eeg_visual_classification_original/TransGAN-master/cifar_checkpoint \
--lstm_path /home/d.sorge/eeg_visual_classification/eeg_visual_classification_main/lstm_256_subject0_epoch_60.pth \
--exp_name cifar_eeg_lstm_train
