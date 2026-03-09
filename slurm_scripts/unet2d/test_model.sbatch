#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=2dunet_cross_entropy_loss_baseline_testing
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:turing:1
#SBATCH --output /home/s3758869/acdc_challenge/slurm_outputs/2dunet_cross_entropy_loss_baseline_testing%J.log

nvidia-smi

export CUDA_VISIBLE_DEVICES=0

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

eval "$(/home/s3758869/miniconda3/bin/conda shell.bash hook)"
conda activate acdc_py37

which python
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

cd /home/s3758869/acdc_challenge/acdc_segmenter
python evaluate_patients.py acdc_logdir/unet2d_cross_entropy_loss_baseline_training