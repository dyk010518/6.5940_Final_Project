#!/bin/bash
#SBATCH -o slurm_logs/resnet_fruits_vegetables.sh.log-%j
#SBATCH --gres=gpu:volta:1

# # Train baseline
# python train_resnet.py 

# # Test baseline
# python test_resnet.py

echo What is your image called without .jpg?

read image_path

# Segment an image
python demo_sam_model.py --image_path assets/fig/$image_path.jpg

# Get classifications on segmented images based on cutoff
python segments_classification_filter.py --segement_directory_path assets/demo/$image_path/ --result_directory_path assets/results/$image_path/

echo Your results are ready! Check them at assets/results/$image_path