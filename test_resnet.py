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

def is_grayscale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

def find_grayscale_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if is_grayscale(os.path.join(directory, filename)):
                print(filename)

class SingleFolderDataset(Dataset):
    def __init__(self, img_dir, target_class, transform=None):
        self.img_dir = Path(img_dir)
        self.img_paths = list(self.img_dir.glob('*.jpg'))  # adjust as necessary if you have other file types
        self.transform = transform
        self.target_class = target_class

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)  # This returns a PIL Image
        if self.transform:
            img = self.transform(img)
        return img, self.target_class


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



def main(path_model, path_classes, path_test_dataset, device, num_classes=0):
    target_model = load_resnet_model(path_model, num_classes, device)

    # Define your transform
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Added this line to handle images with alpha channel
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    categories = read_txt(path_classes)
    
    accuracies = {}

    target_model.eval()
    for category in categories:
        accuracies[category] = {}
        target_index = categories.index(category)

        # Get the data
        folder_name = f"{category}"
        dataset = SingleFolderDataset(f'{path_test_dataset}/{folder_name}', target_class=target_index, transform=data_transforms)  # or 1 for 'phoning'
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = target_model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print(f'Accuracy of the network on {folder_name} test images: {100  * correct / total }')
        accuracies[category] = {'accuracy': 100 * correct / total, 'correct': correct,'count': total}
    
    return accuracies


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


def compute_average_acc(accuracies, path_list_verbs):
    list_verbs = read_txt(path_list_verbs)
    count_total = 0
    correct_total = 0
    for target in list_verbs:
        correct_total += accuracies[target]['correct']
        count_total += accuracies[target]['count']
    
    print(correct_total / count_total)
    return correct_total / count_total


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model', type=str, default='saved_models/fruit_vegetables/baseline/model.pt')
    parser.add_argument('--path_test_dataset', type=str, default='assets/datasets/test')
    parser.add_argument('--path_list_classes', type=str, default='assets/datasets/classes.txt')
    parser.add_argument('--path_result', type=str, default='saved_models/fruit_vegetables/baseline/results')
    parser.add_argument('--num_classes', type=int, default=36)

    args = parser.parse_args()
    path_model = Path(args.path_model)
    path_test_dataset = Path(args.path_test_dataset)
    path_list_classes = Path(args.path_list_classes)
    num_classes = args.num_classes
    path_result = Path(args.path_result)

    device = 'cuda'

    accuracies_main = main(path_model, path_list_classes, path_test_dataset, device, num_classes)
    average_acc = compute_average_acc(accuracies_main, path_list_classes)

    if path_result is not None:
        df = pd.DataFrame(accuracies_main).transpose()

        if not path_result.exists():
            path_result.mkdir(parents=True)
            
        df.to_csv(path_result / 'accuracies_main.csv')
        with open(path_result / 'accuracy.txt', 'w') as f:
            f.write(str(average_acc))