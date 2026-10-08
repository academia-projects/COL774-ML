import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import preprocessor as pre
import pickle
import argparse

'''EXPECTED SIZES :
x -> m*n                (m features and n examples)
y -> 1*n
z -> size_arbitrary*n
weights -> this_z*prev_z*layer_num

'''
np.random.seed(0)

def activation_func(z : np.array, final_layer : bool) :
    if final_layer :
        zp = np.exp(z)
        return zp / np.sum(zp, axis = 0, keepdims=True)
    else :
        return 1/(1 + np.exp(-z))
    
def prediction(num_layers : int , x : np.array , weights : list, biases : np.array) :
    input = x
    for i in range(num_layers) :
        input = activation_func(weights[i]@input + biases[i], i == num_layers - 1)
    return input

def loss(x : np.array , y : np.array , num_layers : int , weights : list , biases : np.array) :
    # pred = prediction(num_layers, x, weights, biases)
    # loss = np.sum(y*pred) + np.sum((1 - y)*(1 - pred))
    # return -loss/y.shape[0]
    pass

def update_weight(x: np.array, y: np.array, num_layers: int, weights: list, biases: list, alpha: np.float64):
    prediction_layer = [x]
    for i in range(num_layers):
        prediction_layer.append(activation_func(weights[i] @ prediction_layer[-1] + biases[i], (i == num_layers - 1)))

    pre_gradient = (-1 / x.shape[1]) * (y - prediction_layer[-1])  # g'(z(n))

    for i in range(num_layers - 1, -1, -1):

        gradient_w = pre_gradient @ prediction_layer[i].T
        gradient_b = np.sum(pre_gradient, axis=1, keepdims=True)
        
        # the trick to not expand to tensor is 
        # pre_gradient is the same as delta or loss commonly used in back propagation, and takes un-updated weights !!!! (stuck for 2 days)
        pre_gradient = (weights[i].T @ pre_gradient) * (prediction_layer[i] * (1 - prediction_layer[i]))
        weights[i] -= alpha * gradient_w
        biases[i] -= alpha * gradient_b
        

    # This is the code for back propagation
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required = True, help='root directory containing the 8 subfolders')
    parser.add_argument('--save_weights_path', type=str, required = True, help='location to save the weights')
    args = parser.parse_args()
    # Define the dimensions of each layer
    layer_dims = [625, 512, 256, 128, 8]

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
    dataset = pre.CustomImageDataset(root_dir=root_dir, csv = csv, transform=pre.numpy_transform)

    # Create the DataLoader
    dataloader = pre.DataLoader(dataset, batch_size=256)

    # Iterate through the DataLoader
    for i in range(15) :
        for images, labels in dataloader:
            # print(images.shape)
            labels = np.eye(8)[labels]      # one hot encoding
            # print(labels.shape)
            update_weight(images.T, labels.T, 4, weights, biases, np.float64(0.001))
            # out = prediction(4, images.T, weights, biases)
        # print(f"Epoch {i + 1} completed in {time.time() - start} seconds")
    # print(weights)

    dict = {}
    dict['weights'] = {}
    dict['bias'] = {}
    for i in range(len(weights)) :
        dict['weights'][f'fc{i + 1}'] = weights[i].T
        dict['bias'][f'fc{i + 1}'] = biases[i].reshape(-1)

    with open(args.save_weights_path + 'weights.pkl', 'wb') as f :
        pickle.dump(dict, f)
