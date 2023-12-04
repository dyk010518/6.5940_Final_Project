import argparse
import os
import json
from pathlib import Path
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import sys
sys.path.insert(1, str(Path.cwd()))
from PIL import Image
from classify_image import main as image_classifier


import shutil


def main(segement_directory_path, result_directory_path, cutoff = 0):
    # we take the directory of the segmented images

    # iterate through the segmented images in the directory
    # run classify_image.py on each image, and if score > cutoff, save the image in the result directory where it's named with the classification label

    # that directory
    path_model = 'saved_models/fruit_vegetables/baseline/model.pt'
    path_classes = 'assets/datasets/classes.txt'
    device = 'cuda'
    num_classes = 36
    for filename in os.listdir(segement_directory_path):
        path_image = os.path.join(segement_directory_path, filename)
        index, max_num, categories = image_classifier(path_model, path_classes, path_image, device, num_classes)
        if max_num.item() > cutoff:
            label = categories[index.item()]
            result_image = os.path.join(result_directory_path, label + "_" + filename)
            shutil.copy(path_image, result_image)





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--segement_directory_path', type=str, default='assets/demo/example_2/')
    parser.add_argument('--result_directory_path', type=str, default='assets/results/example_2/')
    parser.add_argument('--cutoff', type=int, default=0)

    args = parser.parse_args()
    segement_directory_path = Path(args.segement_directory_path)
    result_directory_path = Path(args.result_directory_path)
    cutoff = 0

    try:
        os.makedirs(result_directory_path)
    except:
        None

    main(segement_directory_path, result_directory_path, cutoff)