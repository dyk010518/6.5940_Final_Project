from __future__ import print_function, division

import argparse 
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path 
import random 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

import warnings
warnings.filterwarnings("ignore")

def train(path_dataset: Path, path_save: Path, args: argparse.Namespace):

    LR = args.lr
    MOMENTUM = args.momentum
    STEP_SIZE = args.step_size
    GAMMA = args.gamma
    NUM_EPOCHS = args.num_epochs

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Added this line to handle images with alpha channel
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Added this line to handle images with alpha channel
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {'train': datasets.ImageFolder(path_dataset / 'train',
                                            data_transforms['train']),
                    'val': datasets.ImageFolder(path_dataset / 'val',
                                            data_transforms['val'])
                    }
     
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                        shuffle=True, num_workers=8)
                for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    num_classes = len(image_datasets['train'].classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        metadata = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and valing phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to val mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics 
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                metadata.append(f'Epoch {epoch} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            scheduler.step()

                
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_acc, metadata


    model_conv = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    for param in model_conv.parameters():
        param.requires_grad = False
    
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)  # replace the last FC layer
    model_conv = model_conv.to(device)

    # Observe that all parameters are being optimized
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=STEP_SIZE, gamma=GAMMA)

    model_conv, best_acc, metadata = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=NUM_EPOCHS)

    
    folders = str(path_dataset).split('/')
    name_dataset = '-'.join(folders[-2:])

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    torch.save(model_conv.state_dict(), path_save / 'model.pt')

    with open(path_save / 'metadata_training.txt', 'w') as f:
        for line in metadata:
            f.write(f'{line}\n')
    with open(path_save / 'results.txt', 'a') as f:
        f.write(f'{name_dataset}: {best_acc}\n')

if __name__ == '__main__':

    LR = 0.01
    MOMENTUM = 0.9
    STEP_SIZE = 7
    GAMMA = 0.1
    NUM_EPOCHS = 25
    SEED = 0

    parser = argparse.ArgumentParser(description='Settings for training')
    parser.add_argument("--dataset", type=str, default='train_val_split/', help="dataset name")
    parser.add_argument("--save_folder", type=str, default='fruit_vegetables/baseline/', help="name of the saved model")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="momentum")
    parser.add_argument("--step_size", type=int, default=STEP_SIZE, help="step size")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="gamma")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="number of epochs")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")

    random.seed(SEED)
    torch.manual_seed(SEED)

    args = parser.parse_args()

    path_root = Path.cwd() / 'assets' / 'datasets' 
    path_dataset = path_root / args.dataset
    path_save = Path(f'saved_models/{args.save_folder}')

    print(f'Training on {args.dataset}')
    train(path_dataset, path_save, args)