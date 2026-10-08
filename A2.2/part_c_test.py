import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import argparse
import pickle

torch.manual_seed(0)

transform = transforms.Compose([
    transforms.Resize((64,128)),
    #transforms.CenterCrop(112),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.8898], std=[0.1327])
                  
])
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)


        return image

class CNN(nn.Module):


    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1a_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1a_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2a_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2a_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()
        self.FC1 = nn.Linear(in_features=59520, out_features=512)
        self.FC2 = nn.Linear(in_features=512, out_features=8)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        temp = x
        x = self.conv1a_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1a_2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = temp + x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        temp = x
        x = self.conv2a_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2a_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = temp + x

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        

        x = x.view(x.size(0), -1)
        
        x = self.FC1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.FC2(x)

        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN part b')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='Path to the root of the testing dataset')
    parser.add_argument('--load_weights_path', type=str, required=True, help='Path where the trained model weights are saved')
    parser.add_argument('--save_predictions_path', type=str, required=True, help='Parth where the predictions will be stored')

    args = parser.parse_args()


    root_dir = args.test_dataset_root
    root_weights_path = args.load_weights_path
    predictions_path = args.save_predictions_path
    
    csv_path = os.path.join(root_dir, "public_test.csv")


    dataset = CustomImageDataset(root_dir=root_dir, csv = csv_path, transform=transform)

   
    dataloader = DataLoader(dataset, batch_size=1) 

    model = CNN()
    model.load_state_dict(torch.load(root_weights_path))
    final_predictions = []

    
    for idx, (images) in enumerate(dataloader):
        
        predictions = model(images)
        predictions = torch.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        final_predictions.extend(p.item() for p in predictions)
        
    
    final_predictions = np.array(final_predictions)
    #print(final_predictions)

    with open(predictions_path + 'prediction.pkl', 'wb') as f:
        pickle.dump(final_predictions, f)