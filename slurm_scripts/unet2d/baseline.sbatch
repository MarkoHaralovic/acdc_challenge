#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=2dunet_cross_entropy_loss_baseline_training
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/acdc_challenge/slurm_outputs/2dunet_cross_entropy_loss_baseline_training%J.log

source /home/s3758869/miniconda3/bin/activate
conda activate acdc_py37

cd /home/s3758869/acdc_challenge/acdc_segmenter
python train.py \
  --config unet2D_bn_xent.py