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

torch.manual_seed(0)


transform = transforms.Compose([
    transforms.Resize((50, 100)),  # Resize to 50x100 (height x width)
    transforms.ToTensor(),         # Convert the image to a tensor and also rescales the pixels by dividing them by 255
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
        label = row["class"]

        if self.transform:
            image = self.transform(image)


        return image, label


class CNN(nn.Module):

  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.relu = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.FC1 = nn.Linear(19200, 1)

  
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool1(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool2(x)
    

    x = x.view(x.size(0), -1)
    #print(x.shape)

    x = self.FC1(x)
    

    return x
  

model = CNN()
model = model.float()

criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN part a')
    parser.add_argument('--train_dataset_root', type=str, required=True, help='Path to the root of the training dataset')
    parser.add_argument('--save_weight_path', type=str, required=True, help='Path where the trained model weights will be saved')
    args = parser.parse_args()

    
    root_dir = args.train_dataset_root
    root_weights_path = args.save_weights_path
    csv_path = os.path.join(root_dir, "public_train.csv")

   
    dataset = CustomImageDataset(root_dir=root_dir, csv = csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128)

    

    final_images = []
    final_labels = []
    n=0

    for idx, (images, labels) in enumerate(dataloader):
       final_images.append(images.float())
       final_labels.append(labels.float())
       n+=images.shape[0]

    
    
    epochs = 8
    for i in range(epochs):
        
        
        for j in range(len(final_images)):
            optimizer.zero_grad()
            predictions = model(final_images[j])
            loss = criteria(predictions.squeeze(1), final_labels[j])
            loss.backward()
            optimizer.step()
        
        #print(f'Epoch [{i+1}/{epochs}], Loss: {loss.item()}')
        torch.save(model.state_dict(), root_weights_path + "part_a_binary_model.pth")
    
    

    
    

        
