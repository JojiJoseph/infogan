#!/bin/bash

#SBATCH --job-name=mnist
#SBATCH --output=out_mnist.txt
#SBATCH --gres=gpu:4
#SBATCH --partition=cl1_all_4G

srun python train_mnist.py
