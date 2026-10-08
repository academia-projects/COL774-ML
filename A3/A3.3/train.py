#IMPORTING LIBRARIES

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import gc
import time
import pickle
import numpy as np
import pandas as pd
import random
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import sys


start = time.time()

#SETTING THE SEEDS

# Set a seed value
seed = 0

# Set the seed for PyTorch
torch.manual_seed(seed)

# If using a GPU, set the seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    #torch.manual_seed_all(seed)  # For multiple GPUs

# Set the seed for Python's built-in random module
random.seed(seed)

# Set the seed for NumPy
np.random.seed(seed)

# Make CUDA operations deterministic (if using CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x

def googlenet():
    return GoogleNet()


class CIFAR100TrainDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of tuples (tensor_features, integer_label).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves the sample at the given index."""
        image, label = self.data[idx]
        
        # # if the train _ transform expects pil images
        # image = to_pil_image(image)

        if self.transform:
            image = self.transform(image)
        return image, label
        
#print("Train_dataset_Class ready")

#CUSTOM DATASET FOR TEST

class CIFAR100TestDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of tuples (tensor_features, integer_example_ID).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves the sample at the given index."""
        image, example_id = self.data[idx]
        
        # # if the test transform expects PIL images
        # image = to_pil_image(image)
        
        if self.transform:
            image = self.transform(image)
        return image, example_id


if __name__ == "__main__":

    file_name = sys.argv[0]
    train_set_path = sys.argv[1]
    alpha = sys.argv[2]
    gamma = sys.argv[3]

    with open(train_set_path, 'rb') as train_file:
        total_train_set = pickle.load(train_file)

    # print(type(total_train_set), " is the type of train or test set variable")
    # print(type(total_train_set[0]), " is the type of every elem in train or test set")
    # print(type(total_train_set[0][0]), " is the type of first elem in train or test set elem tuple")
    # print(type(total_train_set[0][1]), " is the type of first elem in train or test set elem tuple")

    # print(len(total_train_set), " is the size of the train set before splitting into val set")
    # print(len(test_set), " is the size of the test set")

    total_train_X = [total_train_set[i][0] for i in range(len(total_train_set))]
    total_train_Y = [total_train_set[i][1] for i in range(len(total_train_set))]

    #test_X = [test_set[i][0] for i in range(len(test_set))]

    #REMEMBER THAT THE TEST_Y ARE NOT THE LABELS BUT THE EXAMPLE IDS SINCE THE TEST CANNOT HAVE LABELS
    #test_Y = [test_set[i][1] for i in range(len(test_set))]

    #print(total_train_X[0].shape, " is the shape of each image i.e. X tensor")
    #train_X is the list of X_tensors in training set
    #train_Y is the list of corresponding Y labels of training set
    #equivalent for test_X and test_Y


    """ MAKING THE TRAINING VALIDATION SPLIT"""
    split = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=42)
    train_indices, val_indices = next(iter(split.split(np.zeros(len(total_train_Y)), total_train_Y)))
    train_indices = set(train_indices)
    val_indices = set(val_indices)
    #print(train_indices, " are the indices of the training set")
    #print(val_indices, " are the indices of the validation set")

    train_size = 36000
    val_size = 4000
    total_size = train_size + val_size  # 40000
    # Calculate the proportion for validation
    validation_ratio = val_size / total_size  # 4000 / 40000 = 0.1

    val_set = [total_train_set[i] for i in val_indices]
    train_set = [total_train_set[i] for i in train_indices]

    train_X = [train_set[i][0] for i in range(len(train_set))]
    train_Y = [train_set[i][1] for i in range(len(train_set))]

    val_X = [val_set[i][0] for i in range(len(val_set))]
    val_Y = [val_set[i][1] for i in range(len(val_set))]


    # print(len( Counter(total_train_Y) ), " is just 100")
    # print(len( Counter(val_Y) ), " is just 100")
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


    train_dataset = CIFAR100TrainDataset(train_set, transform = train_transform)
    #print("train dataset initialised")
    val_dataset = CIFAR100TrainDataset(val_set, transform = test_transform)
    #print("val dataset initialised")
    #test_dataset = CIFAR100TestDataset(test_set, transform = test_transform)
    #print("test dataset initialised")

    batch_size = 200
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )

    #print("train, test, val dataloaders ready")

    net = GoogleNet()
    num_classes = 100
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.Adam(net.parameters(), lr=5e-4,weight_decay=5e-4)

    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,              # Initial learning rate
        momentum=0.9,        # Nesterov momentum
        weight_decay=5e-4,   # Weight decay for regularization
        nesterov=True        # Enable Nesterov momentum
    )

    # Define the learning rate scheduler for 100 epochs
    scheduler_100_epochs = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 90],  # Epochs to divide learning rate
        gamma=0.2                 # Divide by 5
    )

    #MOVING THE MODEL TO GPU IF AVAILABLE 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If multiple GPUs are available, use DataParallel to distribute the workload
    if torch.cuda.device_count() > 1:
        #print(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

    # Move the model to the available devices
    net.to(device)

    def train(epoch):
        #print(f'\nEpoch: {epoch}')
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            break
            # if batch_idx % 100 == 0:
            #     print(f'Batch {batch_idx}/{len(train_loader)} | '
            #           f'Loss: {train_loss/total:.3f} | '
            #           f'Acc: {100.*correct/total:.3f}% ({correct}/{total})')


    #THIS IS ACTUALLY DONE ON THE VALIDATION SET WHILE TRAINING
    def test(epoch):
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # print(f'Test Loss: {test_loss/total:.3f} | '
        #     f'Test Acc: {100.*correct/total:.3f}% ({correct}/{total})')
        return test_loss

    best_val_loss = 1e10
    net.train()
    total_epochs = 100
    for epoch in range(1, total_epochs+1):
        train(epoch)
        curr_val_loss = test(epoch)
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            torch.save(net.state_dict(), 'model.pth')
        scheduler_100_epochs.step()

    

    end = time.time()