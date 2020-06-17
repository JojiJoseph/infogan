#!/bin/bash

#SBATCH --job-name=fashion
#SBATCH --output=out_fashion.txt
#SBATCH --gres=gpu:4
#SBATCH --partition=cl1_all_4G

srun python train_fashion.py
