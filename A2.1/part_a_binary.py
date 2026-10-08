import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import pickle
import argparse

'''EXPECTED SIZES :
x -> m*n                (m features and n examples)
y -> 1*n
z -> size_arbitrary*n
weights -> this_z*prev_z*layer_num

'''
np.random.seed(0)

class CustomImageDataset:
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L") #Convert image to greyscale
        label = row["class"]

        if self.transform:
            image = self.transform(image)

        return np.array(image), label

# Transformations using NumPy
def resize(image, size):
    # return np.array(Image.fromarray(image).resize(size))
    return np.array(image.resize(size))

def to_tensor(image):
    return image.astype(np.float32) / 255.0

def numpy_transform(image, size=(25, 25)):
    image = resize(image, size)
    image = to_tensor(image)
    image = image.flatten()
    return image

class DataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

    def __iter__(self):
        self.start_idx = 0
        return self
    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __next__(self):
        if self.start_idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.start_idx:end_idx]
        images = []
        labels = []

        for idx in batch_indices:
            image, label = self.dataset[idx]
            images.append(image)
            labels.append(label)

        self.start_idx = end_idx

        # Stack images and labels to create batch tensors
        batch_images = np.stack(images, axis=0)
        batch_labels = np.array(labels)

        return batch_images, batch_labels

def activation_func(z : np.array) :
    return 1/(1 + np.exp(-z))

    
def prediction(num_layers : int , x : np.array , weights : list, biases : np.array) :
    input = x
    for i in range(num_layers) :
        input = activation_func(weights[i]@input + biases[i])
    return input

def loss(x : np.array , y : np.array , num_layers : int , weights : list , biases : np.array) :
    pred = prediction(num_layers, x, weights, biases)
    loss = np.sum(y*pred) + np.sum((1 - y)*(1 - pred))
    return -loss/y.shape[0]

def update_weight(x: np.array, y: np.array, num_layers: int, weights: list, biases: list, alpha: np.float64):
    y = y.reshape((1, y.shape[0]))
    prediction_layer = [x]
    for i in range(num_layers):
        prediction_layer.append(activation_func(weights[i] @ prediction_layer[-1] + biases[i]))

    pre_gradient = (-1 / x.shape[1]) * (y - prediction_layer[-1])  # g'(z(n))

    for i in range(num_layers - 1, -1, -1):

        gradient_w = pre_gradient @ prediction_layer[i].T
        gradient_b = np.sum(pre_gradient, axis=1, keepdims=True)
        
        # the trick to not expand to tensor is just handle the n with dimension matching and it all works out (prove later why)
        # pre_gradient is the same as delta or loss commonly used in back propagation, and takes un-updated weights !!!! (stuck for 2 days)
        pre_gradient = (weights[i].T @ pre_gradient) * prediction_layer[i] * (1 - prediction_layer[i])
        weights[i] -= alpha * gradient_w
        biases[i] -= alpha * gradient_b
    # This is the code for back propagation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required = True, help='root directory containing the 8 subfolders')
    parser.add_argument('--save_weights_path', type=str, required = True, help='location to save the weights')
    args = parser.parse_args()

    # Define the dimensions of each layer
    layer_dims = [625, 512, 256, 128, 1]

    # Initialize the weights with zeros
    weights = [np.random.randn(layer_dims[i - 1], layer_dims[i]).astype(np.float64).T * np.sqrt(2/(layer_dims[i - 1]), dtype=np.float64) for i in range(1, len(layer_dims))]
    biases = [np.zeros((layer_dims[i], 1), dtype = np.float64) for i in range(1, len(layer_dims))]

    # Root directory containing the 8 subfolders
    root_dir = args.dataset_root
    mode = 'train' #Set mode to 'train' for loading the train set for training. Set mode to 'val' for testing your model after training. 

    if mode == 'train':
        csv = os.path.join(root_dir, "train.csv")

    elif mode == 'val':
        csv = os.path.join(root_dir, "val.csv")

    # Create the custom dataset
    dataset = CustomImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=256)

    # Iterate through the DataLoader
    for i in range(15) :
        for images, labels in dataloader:
            # print(images.shape)
            # print(labels.shape)
            update_weight(images.T, labels.T, 4, weights, biases, np.float64(0.001))
            # out = prediction(4, images.T, weights, biases)
        print(f"Epoch {i + 1} completed ")      #in {time.time() - start} seconds
    print(weights)

    dict = {}
    dict['weights'] = {}
    dict['bias'] = {}
    for i in range(len(weights)) :
        dict['weights'][f'fc{i + 1}'] = weights[i].T
        dict['bias'][f'fc{i + 1}'] = biases[i].reshape(-1)

    with open(args.save_weights_path + '/weights.pkl', 'wb') as f :
        pickle.dump(dict, f)
