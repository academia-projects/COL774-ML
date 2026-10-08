import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
#from torch.cuda.amp import GradScaler, autocast
import gc
import time
import pickle

import argparse
import sys

start = time.time()

from torchvision.transforms.functional import to_pil_image


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
        
        #the test transform expects PIL images
        image = to_pil_image(image)
        
        
        if self.transform:
            image = self.transform(image)
        return image, example_id
if __name__ == "__main__":

    file_name = sys.argv[0]
    torch_model_path = sys.argv[1]
    test_file_path = sys.argv[2]
    alpha = sys.argv[3]
    gamma = sys.argv[4]

    with open(test_file_path, 'rb') as test_file:
        test_set = pickle.load(test_file)

    #print(len(test_set))

    test_X = [test_set[i][0] for i in range(len(test_set))]
    #print(test_X[0].shape)

    #test_Y are the IDS of each example, THEY ARE NOT THE LABELS
    test_Y = [test_set[i][1] for i in range(len(test_set))]
    #print(type ( test_Y[0] ) )
    #print(test_Y[0], test_Y[1], test_Y[2])

    #CUSTOM DATASET FOR TEST

    #print("test_dataset_class_ready")

    cifar100_mean = [0.5071, 0.4867, 0.4408]
    cifar100_std = [0.2675, 0.2565, 0.2761]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    #print("test_transform", "ready")

    test_dataset = CIFAR100TestDataset(test_set, transform=test_transform)

    batch_size = 128

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    net = models.densenet121(weights=None)
    num_classes = 100
    num_features = net.classifier.in_features

    net.classifier = nn.Linear(num_features, 100)

    net.features.conv0 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    net.features.pool0 = nn.Identity()
    
    
    #state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    #print(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net.load_state_dict(torch.load('model.pth', map_location = device))

    net.eval()

    #print("evaluation mode i.e. testing mode")

    final_outs = []
    #print(type(final_outs))

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=2, dim=1)

            max_probs = top_probs[:, 0]            # Maximum probabilities
            second_max_probs = top_probs[:, 1]     # Second maximum probabilities
            max_indices = top_indices[:, 0]        # Indices of maximum probabilities

            # Compute the ratio
            ratio = max_probs / second_max_probs

            # Prepare a tensor of -1s with the same type and device as max_indices
            negative_ones = torch.full_like(max_indices, -1)

            # Apply the condition
            #predicted = torch.where(ratio > 25, max_indices, negative_ones)
            predicted = torch.where(max_probs > 0.901, max_indices, negative_ones)
            
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            final_outs.append(predicted)

    final_outs = torch.cat(final_outs)
    #print(final_outs)

    test_Y = torch.tensor(test_Y)
    #print(test_Y)

    final_outs = final_outs.cpu()


    outputs_with_ids = torch.stack([test_Y, final_outs])
    #print(outputs_with_ids)
    np_output = outputs_with_ids.numpy()
    #print(np_output)
    np_output = np_output.T
    #print(np_output)

    import pandas as pd
    df = pd.DataFrame(np_output, columns = ["ID", "Predicted_label"])

    #print(df)

    df.to_csv('submission.csv', index=False, header=True)

    end = time.time()
    #print(end - start)
