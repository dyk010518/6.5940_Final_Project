#!/bin/bash
#SBATCH -o slurm_logs/resnet_fruits_vegetables.sh.log-%j
#SBATCH --gres=gpu:volta:1

# Train baseline
# python train_resnet.py 

# Test baseline

python test_resnet.py
