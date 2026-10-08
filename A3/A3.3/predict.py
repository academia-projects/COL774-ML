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
from torch.optim import LBFGS



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
    torch_model_path = sys.argv[1]
    test_file_path = sys.argv[2]
    alpha = sys.argv[3]
    gamma = sys.argv[4]

    with open(test_file_path, 'rb') as test_file:
        test_set = pickle.load(test_file)

    test_X = [test_set[i][0] for i in range(len(test_set))]

    #REMEMBER THAT THE TEST_Y ARE NOT THE LABELS BUT THE EXAMPLE IDS SINCE THE TEST CANNOT HAVE LABELS
    test_Y = [test_set[i][1] for i in range(len(test_set))]

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    batch_size = 200

    test_transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    test_dataset = CIFAR100TestDataset(test_set, transform = test_transform)
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

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

    #LOADING THE SAVED MODEL

    net.load_state_dict(torch.load(torch_model_path, map_location = device))
    net.eval()
    net=net.to(device)
    predicted_val_Y = []
    predicted_test_Y = []
    val_Y_probs = []
    test_Y_probs = []
    val_logits = []
    test_logits = []
   

    #pratyush
    #class_wise_threshold_vals = [0.26601114869117737, 0.7094566226005554, 0.8718365430831909, 0.8966909050941467, 1, 0.7869625687599182, 0.1622653752565384, 0.5349425673484802, 0.10160939395427704, 0.3468324840068817, 0.8302435278892517, 0.4847811162471771, 0.6543588042259216, 0.7175024151802063, 0.2790333926677704, 0.6274416446685791, 0.41219356656074524, 0.19550760090351105, 0.8954567909240723, 0.460099458694458, 0.267780601978302, 0.5528279542922974, 0.5473291277885437, 0.2176632583141327, 0.2131403237581253, 0.6733903884887695, 0.5333231687545776, 0.41273388266563416, 0.24945034086704254, 0.586205244064331, 0.34926220774650574, 0.526416540145874, 0.5779319405555725, 0.48514246940612793, 0.3526076376438141, 1, 0.19569514691829681, 0.4929760694503784, 0.672088086605072, 0.14699727296829224, 0.540529727935791, 0.176728293299675, 0.42769792675971985, 0.22791099548339844, 0.9950591325759888, 0.4411543905735016, 0.8818264007568359, 0.40428072214126587, 0.21055461466312408, 0.5750008225440979, 0.8072423338890076, 0.6030920743942261, 0.8491771817207336, 0.45764029026031494, 0.34853649139404297, 0.8699443340301514, 0.3052920699119568, 0.48690444231033325, 0.2134406417608261, 0.3407685160636902, 0.4336067736148834, 0.7935525178909302, 0.6988632082939148, 0.15963353216648102, 0.5620270371437073, 0.5917965173721313, 0.14960968494415283, 0.6686303019523621, 0.1723623424768448, 0.44424983859062195, 0.2409902960062027, 0.800077497959137, 1, 0.7167086601257324, 0.8365378975868225, 0.13120360672473907, 0.12431259453296661, 0.2601095139980316, 0.6055464148521423, 0.36977529525756836, 0.9026542901992798, 0.7813990116119385, 0.37659651041030884, 0.5903930068016052, 0.6286007761955261, 0.3376988172531128, 0.5014891624450684, 0.7475535869598389, 0.1316273808479309, 0.6112165451049805, 0.34133169054985046, 0.4786504805088043, 0.9160618185997009, 0.631229043006897, 0.5205046534538269, 0.5682248473167419, 0.6295484900474548, 0.2268284112215042, 0.9173423647880554, 0.7019006609916687]
    #verma
    #class_wise_threshold_vals = [0.20813632011413574, 0.23151341080665588, 0.7775099277496338, 0.776821494102478, 0.6609290242195129, 0.7771540880203247, 0.21231305599212646, 0.4377904534339905, 0.12031610310077667, 0.4395442008972168, 0.8332125544548035, 0.7489187121391296, 0.4480929374694824, 0.7633393406867981, 0.2676592171192169, 0.8000593781471252, 0.5567203164100647, 0.12497472018003464, 0.7422542572021484, 0.3697696328163147, 0.2015111744403839, 0.5099772810935974, 0.6541310548782349, 0.24609419703483582, 0.1895981878042221, 0.7050914764404297, 0.47708097100257874, 0.48608747124671936, 0.14206339418888092, 0.4028509557247162, 0.470392107963562, 0.8050215840339661, 0.6205552816390991, 0.5131059885025024, 0.6527290344238281, 0.9393340349197388, 0.3873369097709656, 0.8086521625518799, 0.8207855224609375, 0.10529764741659164, 0.5799058675765991, 0.210010826587677, 0.4904945194721222, 0.5151147842407227, 1, 0.3490811288356781, 0.6420263051986694, 0.258931040763855, 0.1666448414325714, 0.566468358039856, 0.8444564938545227, 0.7580419778823853, 0.9176121950149536, 0.43051809072494507, 0.38030698895454407, 1, 0.142306849360466, 0.27017053961753845, 0.1407654583454132, 0.3961467146873474, 0.45610687136650085, 0.7774103879928589, 0.4739784896373749, 0.2543848156929016, 0.665947675704956, 0.8265551328659058, 0.13334141671657562, 0.5141022801399231, 0.25996914505958557, 0.16424520313739777, 0.3531413674354553, 0.7880232930183411, 1, 0.747444212436676, 0.7984039187431335, 0.25857192277908325, 0.15584701299667358, 0.46126076579093933, 0.6644928455352783, 0.37225908041000366, 0.9070372581481934, 0.9315620064735413, 0.12250132858753204, 0.7621548175811768, 0.890531599521637, 0.12427768856287003, 0.6598858833312988, 0.496299684047699, 0.3852955400943756, 0.6431225538253784, 0.5940800905227661, 0.4437951147556305, 0.6238033771514893, 0.3436037600040436, 0.2950720191001892, 0.5842674970626831, 0.5854617357254028, 0.3030117154121399, 1, 0.408031702041626]



    


    #print(class_wise_threshold_vals, " is the class wise threshold probability values")


    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)

            #FOR TEMPERATURE SCALING
            #outputs /= optimal_temperature
            
            probs = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=2, dim=1)

            max_probs = top_probs[:, 0]            # Maximum probabilities
            second_max_probs = top_probs[:, 1]     # Second maximum probabilities
            max_indices = top_indices[:, 0]        # Indices of maximum probabilities

            predicted = torch.full_like(max_indices, -1)
            # for i in range(max_indices.shape[0]):
            #     pred_class = max_indices[i]
            #     pred_max_prob = max_probs[i]

            #     # 0.57 , 0.85 optimal vals without temp. scaling
            #     atleast_this_much = 0.57
            #     atmost_this_much = 0.85
            #     if (   pred_max_prob > max(  atleast_this_much,
            #                             min( class_wise_threshold_vals[pred_class], atmost_this_much )  
            #                             )   ):
            #         predicted[i] = pred_class
            # # # Compute the ratio
            # ratio = max_probs / second_max_probs

            # # Prepare a tensor of -1s with the same type and device as max_indices
            negative_ones = torch.full_like(max_indices, -1)

            # # Apply the condition
            # #predicted = torch.where(ratio > 25, max_indices, negative_ones)
            predicted = torch.where((max_probs > 0.9991), max_indices, negative_ones)
            
            #print(predicted)
            predicted_test_Y.append(predicted)

    predicted_test_Y = torch.cat(predicted_test_Y)
    print(predicted_test_Y, " is the predicted test Y")

    a=torch.tensor(test_Y)
    b=torch.stack([a, predicted_test_Y.cpu()]).numpy().T
    pd.DataFrame(b, columns=["ID", "Predicted_label"]).to_csv('submission.csv', index=False)


