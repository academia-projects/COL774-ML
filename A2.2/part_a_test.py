import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import argparse
import torch
import torch.nn as nn
import pickle

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

        if self.transform:
            image = self.transform(image)


        return image
    
class CNN(nn.Module):

  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.relu = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.FC1 = nn.Linear(in_features=19200, out_features=1)

  
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool1(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool2(x)

    x = x.view(x.size(0), -1)
    
    
    x = self.FC1(x)
    

    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN part a')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='Path to the root of the testing dataset')
    parser.add_argument('--load_weights_path', type=str, required=True, help='Path where the trained model weights are saved')
    parser.add_argument('--save_predictions_path', type=str, required=True, help='Path where the predictions will be stored')

    args = parser.parse_args()


    root_dir = args.test_dataset_root
    root_weights_path = args.load_weights_path
    predictions_path = args.save_predictions_path
    csv_path = os.path.join(root_dir, "public_test.csv")


    dataset = CustomImageDataset(root_dir=root_dir, csv = csv_path, transform=transform)


    dataloader = DataLoader(dataset, batch_size=1)

    model = CNN()
    model.load_state_dict(torch.load(root_weights_path))
    
    final_predictions=[]

    for idx, (images) in enumerate(dataloader):
        predictions = model(images)
        final_predictions.extend(p.item() for p in predictions)

    
    final_predictions = np.array(final_predictions)
    final_predictions = np.where(final_predictions>0, 1, 0)
    #print(final_predictions)
    

    with open(predictions_path + 'prediction.pkl', 'wb') as f:
        pickle.dump(final_predictions, f)
    






