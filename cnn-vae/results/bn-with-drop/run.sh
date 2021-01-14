#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

#SBATCH --mail-user=harvey@dal.ca
#SBATCH --mail-type=ALL

#source ~/pytorch/bin/activate
conda init bash
source ~/.bashrc
conda activate ~/venvs/pytorch
python3 ./main.py
