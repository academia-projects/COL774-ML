import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import time
import argparse

np.random.seed(0)

# given code starts here -------------------------------------------------
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

class TestImageDataset:
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

        if self.transform:
            image = self.transform(image)

        return np.array(image)

class TestDataLoader:
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
            image = self.dataset[idx]
            images.append(image)

        self.start_idx = end_idx

        # Stack images and labels to create batch tensors
        batch_images = np.stack(images, axis=0)

        return batch_images

class TrainImageDataset:
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

class TrainDataLoader:
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
# given code ends here ---------------------------------------------------

def activation_func(z : np.array, final_layer : bool) :
    if final_layer :
        zp = np.exp(z)
        return zp / np.sum(zp, axis = 0, keepdims=True)
    else :
        return 1/(1 + np.exp(-z))
    
def prediction(num_layers : int , x : np.array , weights : list, biases : np.array) :
    input = x
    for i in range(num_layers) :
        input = activation_func(weights[i]@input + biases[i], (i == num_layers - 1))
    return input

def accuracy(y : np.array, y_pred : np.array) :
    return np.sum(y == y_pred)

def loss(x : np.array , y : np.array , num_layers : int , weights : list , biases : np.array) :
    pred = prediction(num_layers, x, weights, biases)
    cross_entropy = -np.sum(y * np.log(pred))
    return cross_entropy

def update_weight(x: np.array, y: np.array, num_layers: int, weights: list, biases: list, alpha: np.float64, moment_weights: list, variance_weights: list, moment_biases: list, variance_biases: list, beta1: np.float64, beta2: np.float64, epsilon: np.float64, epoch : int) :
            prediction_layer = [x]
            for i in range(num_layers):
                prediction_layer.append(activation_func(weights[i] @ prediction_layer[-1] + biases[i], (i == num_layers - 1)))

            pre_gradient = (-1 / x.shape[1]) * (y - prediction_layer[-1])  # g'(z(n))

            for i in range(num_layers - 1, -1, -1):

                gradient_w = pre_gradient @ prediction_layer[i].T
                gradient_b = np.sum(pre_gradient, axis=1, keepdims=True)
                
                pre_gradient = (weights[i].T @ pre_gradient) * (prediction_layer[i] * (1 - prediction_layer[i]))
                # Adam optimization

                moment_weights[i] = beta1 * moment_weights[i] + (1 - beta1) * gradient_w
                variance_weights[i] = beta2 * variance_weights[i] + (1 - beta2) * (gradient_w ** 2)

                moment_biases[i] = beta1 * moment_biases[i] + (1 - beta1) * gradient_b
                variance_biases[i] = beta2 * variance_biases[i] + (1 - beta2) * (gradient_b ** 2)

                alpha_t = alpha * np.power((1 - np.power(beta2, epoch + 1)), 0.5) / (1 - np.power(beta1, epoch + 1))
                epsilon_t = epsilon * np.power((1 - np.power(beta2, epoch + 1)), 0.5)

                weights[i] -= alpha_t * moment_weights[i] / (np.sqrt(variance_weights[i]) + epsilon_t)
                biases[i] -= alpha_t * moment_biases[i] / (np.sqrt(variance_biases[i]) + epsilon_t)
    

if __name__ == '__main__':
    start = time.time()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_root', type=str, required = True, help='root directory containing the 8 subfolders')
    # parser.add_argument('--test_dataset_root', type=str, required = True, help='directory containing the test images')
    # parser.add_argument('--save_weights_path', type=str, required = True, help='location to save the weights')
    # parser.add_argument('--save_predictions_path', type=str, required = False, help='location to save the predictions')
    # args = parser.parse_args()

    # Define the dimensions of each layer
    layer_dims = [625, 512, 256, 128, 32, 8]
    num_layers = len(layer_dims) - 1

    # Initialize the weights with zeros
    weights = [np.random.randn(layer_dims[i - 1], layer_dims[i]).astype(np.float64).T * np.sqrt(2/(layer_dims[i - 1]), dtype=np.float64) for i in range(1, len(layer_dims))]
    biases = [np.zeros((layer_dims[i], 1), dtype = np.float64) for i in range(1, len(layer_dims))]

    # Root directory containing the 8 subfolders
    root_dir = "dataset_for_A2/multi_dataset"

    # Create the train data
    train_dataset = TrainImageDataset(root_dir=root_dir, csv = os.path.join(root_dir, "train.csv"), transform=numpy_transform)
    train_dataloader = TrainDataLoader(train_dataset, batch_size=256)

    # initialise moments and variances for adam optimization
    moment_weights = [np.zeros_like(weights[i]) for i in range(len(weights))]
    variance_weights = [np.zeros_like(weights[i]) for i in range(len(weights))]

    moment_biases = [np.zeros_like(biases[i]) for i in range(len(biases))]
    variance_biases = [np.zeros_like(biases[i]) for i in range(len(biases))]

    out = []
    beta1 = 0.95
    beta2 = 0.999
    epsilon = 1e-12
    alpha = 1e-3

    dict = {}
    dict['weights'] = {}
    dict['bias'] = {}

    x = []
    y = []
    n = 0 
    for images, labels in train_dataloader:
        x.append(images.T)              # has the transposed form of the images
        y.append(np.eye(8)[labels].T)   # one hot encoding and transposed
        n += images.shape[0]
    
    for i in range(1200) : 
        match = 0
        closs = 0
        for j in range(len(x)):       # creating batches of 256 images
            # labels = np.eye(8)[labels]  # one hot encoding
            update_weight(x[j], y[j], num_layers, weights, biases, alpha, moment_weights, variance_weights, moment_biases, variance_biases, beta1, beta2, epsilon, i)
            predic = prediction(num_layers, x[j], weights, biases)
            predic = np.argmax(predic, axis = 0)
            match += np.sum(np.argmax(y[j], axis = 0) == predic)
            closs += loss(x[j], y[j], num_layers, weights, biases)
        
        if i == 800 :       # 800 was the best
            beta2 = 0.99
        if i == 1100 :
            alpha = 5e-4
            beta2 = 0.9995
        if (i + 1) % 5 == 0 :
            print(f"Epoch {i + 1} completed in {time.time() - start} seconds")
            print('accuracy is ', match/n)
            print('loss is ', closs/n)
            # if match/n < accuracy_init :
            #     continue
            # accuracy_init = match/n
            # for k in range(num_layers) :
            #     dict['weights'][f'fc{k + 1}'] = weights[k].T
            #     dict['bias'][f'fc{k + 1}'] = biases[k].reshape(-1)

            # with open(args.save_weights_path + '/weights.pkl', 'wb') as f :          
            #     pickle.dump(dict, f)
            # print('Weights saved in epoch ', i + 1)
            # print(accuracy_init)

            
    '''# Create the test data
    test_dataset = TestImageDataset(root_dir = root_dir, csv = os.path.join(root_dir, "val.csv"), transform = numpy_transform)
    test_dataloader = TestDataLoader(test_dataset, batch_size=256)

    # Predict the test data
    for idx, images in enumerate(test_dataloader):
        # print(images.shape)  # Should be [batch_size, 625]
        out.extend(np.argmax(prediction(num_layers, images.T, weights, biases), axis = 0).tolist())
        
        #this is for the final output'''
    
    testing_dataset = TrainImageDataset(root_dir = root_dir, csv = os.path.join(root_dir, "val.csv"), transform = numpy_transform)
    testing_dataloader = TrainDataLoader(testing_dataset, batch_size=256)

    out = []
    # actual = []
    for images, labels in testing_dataloader:
        # actual.extend(labels.tolist())
        labels = np.eye(8)[labels]
        out.extend(np.argmax(prediction(num_layers, images.T, weights, biases), axis = 0).tolist())
    
    
    print(out)

    with open('predictions.pkl', 'wb') as f :
        pickle.dump(out, f)

    end = time.time()
    print('time taken : ', end - start)
