#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00

#SBATCH --mail-user=harvey@dal.ca
#SBATCH --mail-type=ALL

#source ~/pytorch/bin/activate
conda init bash
source ~/.bashrc
conda activate ~/venvs/pytorch
python3 ./cnn_vae.py
