import argparse
import os
import json
from pathlib import Path
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import sys
sys.path.insert(1, str(Path.cwd()))

from PIL import Image


def load_resnet_model(path_resnet_model, num_classes, device):
    target_model = models.resnet50()
    target_model.fc = nn.Linear(target_model.fc.in_features, num_classes)

    state_dict = torch.load(path_resnet_model, map_location='cpu')
    target_model.load_state_dict(state_dict)
    target_model = target_model.to(device)

    return target_model

def read_txt(path_txt):
    with open(path_txt, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



def main(path_model, path_classes, path_image, device, num_classes=0):
    target_model = load_resnet_model(path_model, num_classes, device)

    categories = read_txt(path_classes)

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Added this line to handle images with alpha channel
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    target_model.eval()
    image = Image.open(path_image)
    image = data_transforms(image).to(device)

    output = target_model(image.unsqueeze(0))
    output = torch.nn.functional.softmax(output.cpu())
    # print(output)
    # print("Max", torch.max(output).item())
    # print(path_image)
    # print(categories[torch.argmax(output).item()])

    return torch.argmax(output), torch.max(output), categories
    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model', type=str, default='saved_models/fruit_vegetables/baseline/model.pt')
    parser.add_argument('--path_image', type=str, default='assets/demo/example_2/example_2_segment_10.png')
    parser.add_argument('--path_list_classes', type=str, default='assets/datasets/classes.txt')
    parser.add_argument('--num_classes', type=int, default=36)

    args = parser.parse_args()
    path_model = Path(args.path_model)
    path_image = Path(args.path_image)
    path_list_classes = Path(args.path_list_classes)
    num_classes = args.num_classes
    # path_result = Path(args.path_result)

    device = 'cuda'

    label = main(path_model, path_list_classes, path_image, device, num_classes)