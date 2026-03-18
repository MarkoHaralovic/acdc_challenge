#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=2dunet_boundary_aware_cross_entropy_loss_training
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/acdc_challenge/slurm_outputs/2dunet_boundary_aware_cross_entropy_loss_training%J.log

nvidia-smi

export CUDA_VISIBLE_DEVICES=0

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

eval "$(/home/s3758869/miniconda3/bin/conda shell.bash hook)"
conda activate acdc_py37

cd /home/s3758869/acdc_challenge/acdc_segmenter
python train.py \
  --config unet2D_bn_boundary_aware.py \
  --log-dir /home/s3758869/acdc_challenge/acdc_segmenter/acdc_logdir/unet2d_boundary_aware_cross_entropy_loss_training

python evaluate_patients.py acdc_logdir/unet2d_boundary_aware_cross_entropy_loss_training