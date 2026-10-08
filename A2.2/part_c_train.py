import torch    # type: ignore
import torch.nn as nn   # type: ignore
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader    # type: ignore
import torchvision.transforms as transforms # type: ignore
import pandas as pd
import argparse
import matplotlib.pyplot as plt

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
        label = row["class"]

        if self.transform:
            image = self.transform(image)


        return image, label

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


model = CNN()
model = model.float()

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN part b')
    parser.add_argument('--train_dataset_root', type=str, required=True, help='Path to the root of the training dataset')
    parser.add_argument('--save_weight_path', type=str, required=True, help='Path where the trained model weights will be saved')
    args = parser.parse_args()

    
    root_dir = args.train_dataset_root
    root_weights_path = args.save_weight_path
    #test_root_dir = "./dataset_for_A2.2/multi_dataset"
    
    csv_path = os.path.join(root_dir, "public_train.csv")
    #test_csv_path = os.path.join(root_dir, "public_test.csv")
   
    dataset = CustomImageDataset(root_dir=root_dir, csv = csv_path, transform=transform)
    #test_dataset = CustomImageDataset(root_dir=test_root_dir, csv= test_csv_path, transform=transform)
 
    dataloader = DataLoader(dataset, batch_size=128) 
    #test_dataloader = DataLoader(test_dataset, batch_size=800)
    train_images = []
    train_labels = []
    #test_images = []
    #test_labels = []
    #train_loss = []
    #test_accuracy = []
    #epoch= []
    
    n=0

    # for idx, (images, labels) in enumerate(test_dataloader):
    #     test_labels.extend(l.long().item() for l in labels)
    #     test_images.append(images.float())
    

    for idx, (images, labels) in enumerate(dataloader):
        train_images.append(images.float())
        train_labels.append(labels.long())
        n+=images.shape[0]
    
    epochs = 15
    #test_labels = np.array(test_labels)
    for i in range(epochs):
        #final_predictions=[]
        #losses=0
        #epoch.append(i+1)
        for j in range(len(train_images)):
            optimizer.zero_grad()
            predictions = model(train_images[j].float())
            
            loss = criteria(predictions.squeeze(1), train_labels[j].long())
            
            loss.backward()
            optimizer.step()
            
       
        torch.save(model.state_dict(), root_weights_path + "part_c_multi_model.pth")

    
    

