import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pickle

import matplotlib.pyplot as plt
import argparse
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

def activation_func(z : np.array, final_layer : bool) :
    if final_layer :
        zp = np.exp(z)
        return zp / np.sum(zp, axis = 1, keepdims=True)
    else :
        return np.where(z>0, z, 0)

def prediction(num_layers : int , x : np.array , weights : list, biases : np.array) :
    input = x
    for i in range(num_layers) :
        #print(weights[i].shape)
        input = activation_func(input@weights[4-i] + biases[4-i], (i == num_layers - 1))
    return input



def loss(x : np.array , y : np.array , num_layers : int , weights : list , biases : np.array) :
    #print(x.shape)
    pred = prediction(num_layers, x, weights, biases)
    cross_entropy = np.sum(y * np.log(pred))
    return cross_entropy

def relu(x):
    return np.where(x>0, x, 0)

def softmax(x):
    z = np.exp(x)
    return z/np.sum(z, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return np.where(x>0, 1, 0)

def sigmoid_derivative(x):

    return x*(1-x)

def loss_derivative(actual_output, pred_output):
    ld = 0

    for i in range(len(actual_output)):
        if(actual_output[i] == 1):
            ld += -1/pred_output[i]
        else:
            ld += 1/(1 - pred_output[i])   
        
    return ld/len(actual_output)

class NeuralNetwork:

    def __init__(self, input_size, hidden_layer_1_size, hidden_layer_2_size, hidden_layer_3_size, hidden_layer_4_size, output_size, batch_size):
        self.weights_input_hidden1 = (np.random.randn(input_size, hidden_layer_1_size))*np.sqrt(2.0 / (input_size))
        self.weights_hidden1_hidden2 = (np.random.randn(hidden_layer_1_size, hidden_layer_2_size))*np.sqrt(2.0 / (hidden_layer_1_size))
        self.weights_hidden2_hidden3 = (np.random.randn(hidden_layer_2_size, hidden_layer_3_size))*np.sqrt(2.0 / (hidden_layer_2_size))
        self.weights_hidden3_hidden4 = (np.random.randn(hidden_layer_3_size, hidden_layer_4_size))*np.sqrt(2.0 / (hidden_layer_3_size))
        self.weights_hidden4_output = (np.random.randn(hidden_layer_4_size, output_size))*np.sqrt(2.0 / (hidden_layer_4_size))

        self.bias_input_hidden1 = np.zeros(shape=(1, hidden_layer_1_size))
        self.bias_hidden1_hidden2 = np.zeros(shape=(1, hidden_layer_2_size))
        self.bias_hidden2_hidden3 = np.zeros(shape=(1, hidden_layer_3_size))
        self.bias_hidden3_hidden4 = np.zeros(shape=(1, hidden_layer_4_size))
        self.bias_hidden4_output = np.zeros(shape=(1, output_size))

    def forward_prop(self, X):
        #print(X.shape)
        #print((self.weights_input_hidden1).shape)
        #print((self.bias_input_hidden1).shape)
        self.hidden1_result = X@(self.weights_input_hidden1) + self.bias_input_hidden1
        self.hidden2_input = relu(self.hidden1_result)

        self.hidden2_result = self.hidden2_input@(self.weights_hidden1_hidden2) + self.bias_hidden1_hidden2
        self.hidden3_input = relu(self.hidden2_result)

        self.hidden3_result = self.hidden3_input@(self.weights_hidden2_hidden3) + self.bias_hidden2_hidden3
        self.hidden4_input = relu(self.hidden3_result)

        self.hidden4_result = self.hidden4_input@(self.weights_hidden3_hidden4) + self.bias_hidden3_hidden4
        self.output_input = relu(self.hidden4_result)

        self.output_result = self.output_input@(self.weights_hidden4_output) + self.bias_hidden4_output
        self.final_output = softmax(self.output_result)

        return self.final_output

    def back_prop(self, actual_output, learning_rate, X, batch_size, alpha, beta1, beta2, moment_weights, variance_weights, moment_biases, variance_biases, epsilon, epoch):
        #h3_o_delta = loss_derivative(actual_output, self.final_output)*sigmoid_derivative(self.final_output)
        #actual_output = actual_output.reshape(-1,1)
        #print(self.output_input.shape)
        h4_o_delta = (-1/batch_size)*(actual_output - self.final_output)
        gradient_w_h4_o = (self.output_input.T)@h4_o_delta
        gradient_b_h4_o = np.ones(shape=(1, batch_size))@h4_o_delta
        moment_weights[0] = beta1 * moment_weights[0] + (1 - beta1) * gradient_w_h4_o
        moment_biases[0] = beta1 * moment_biases[0] + (1 - beta1) * gradient_b_h4_o
        variance_weights[0] = beta2 * variance_weights[0] + (1 - beta2) * (gradient_w_h4_o**2)
        variance_biases[0] = beta2 * variance_biases[0] + (1 - beta2) * (gradient_b_h4_o**2)

        h3_h4_delta = (h4_o_delta@self.weights_hidden4_output.T)*relu_derivative(self.output_input)
        gradient_w_h3_h4 = (self.hidden4_input.T)@h3_h4_delta
        gradient_b_h3_h4 = np.ones(shape=(1, batch_size))@h3_h4_delta
        moment_weights[1] = beta1 * moment_weights[1] + (1 - beta1) * gradient_w_h3_h4
        moment_biases[1] = beta1 * moment_biases[1] + (1 - beta1) * gradient_b_h3_h4
        variance_weights[1] = beta2 * variance_weights[1] + (1 - beta2) * (gradient_w_h3_h4**2)
        variance_biases[1] = beta2 * variance_biases[1] + (1 - beta2) * (gradient_b_h3_h4**2)
       
        h2_h3_delta = (h3_h4_delta@self.weights_hidden3_hidden4.T)*relu_derivative(self.hidden4_input)
        gradient_w_h2_h3 = (self.hidden3_input.T)@h2_h3_delta
        gradient_b_h2_h3 = np.ones(shape=(1, batch_size))@h2_h3_delta
        moment_weights[2] = beta1 * moment_weights[2] + (1 - beta1) * gradient_w_h2_h3
        moment_biases[2] = beta1 * moment_biases[2] + (1 - beta1) * gradient_b_h2_h3
        variance_weights[2] = beta2 * variance_weights[2] + (1 - beta2) * (gradient_w_h2_h3**2)
        variance_biases[2] = beta2 * variance_biases[2] + (1 - beta2) * (gradient_b_h2_h3**2)

        h1_h2_delta = (h2_h3_delta@self.weights_hidden2_hidden3.T)*relu_derivative(self.hidden3_input)
        gradient_w_h1_h2 = (self.hidden2_input.T)@h1_h2_delta
        gradient_b_h1_h2 = np.ones(shape=(1, batch_size))@h1_h2_delta
        moment_weights[3] = beta1 * moment_weights[3] + (1 - beta1) * gradient_w_h1_h2
        moment_biases[3] = beta1 * moment_biases[3] + (1 - beta1) * gradient_b_h1_h2
        variance_weights[3] = beta2 * variance_weights[3] + (1 - beta2) * (gradient_w_h1_h2**2)
        variance_biases[3] = beta2 * variance_biases[3] + (1 - beta2) * (gradient_b_h1_h2**2)

        i_h1_delta = (h1_h2_delta@self.weights_hidden1_hidden2.T)*relu_derivative(self.hidden2_input)
        gradient_w_i_h1 = (X.T)@i_h1_delta
        gradient_b_i_h1 = np.ones(shape=(1, batch_size))@i_h1_delta
        moment_weights[4] = beta1 * moment_weights[4] + (1 - beta1) * gradient_w_i_h1
        moment_biases[4] = beta1 * moment_biases[4] + (1 - beta1) * gradient_b_i_h1
        variance_weights[4] = beta2 * variance_weights[4] + (1 - beta2) * (gradient_w_i_h1**2)
        variance_biases[4] = beta2 * variance_biases[4] + (1 - beta2) * (gradient_b_i_h1**2)

        alpha_t = alpha * (1 - np.power(0.999, epoch + 1))#np.power((1 - np.power(0.999, epoch + 1)), 0.5) / (1 - np.power(beta1, epoch + 1))
        epsilon_t = epsilon * np.power((1 - np.power(beta2, epoch + 1)), 0.5)

        
        self.weights_hidden4_output -= alpha_t * moment_weights[0] / (np.sqrt(variance_weights[0]) + epsilon_t)
        self.bias_hidden4_output -= alpha_t * moment_biases[0] / (np.sqrt(variance_biases[0]) + epsilon_t)

        self.weights_hidden3_hidden4 -= alpha_t * moment_weights[1] / (np.sqrt(variance_weights[1]) + epsilon_t)
        self.bias_hidden3_hidden4 -= alpha_t * moment_biases[1] / (np.sqrt(variance_biases[1]) + epsilon_t)
        self.weights_hidden2_hidden3 -= alpha_t * moment_weights[2] / (np.sqrt(variance_weights[2]) + epsilon_t)
        self.bias_hidden2_hidden3 -= alpha_t * moment_biases[2] / (np.sqrt(variance_biases[2]) + epsilon_t)

        self.weights_hidden1_hidden2 -= alpha_t * moment_weights[3] / (np.sqrt(variance_weights[3]) + epsilon_t)
        self.bias_hidden1_hidden2 -= alpha_t * moment_biases[3] / (np.sqrt(variance_biases[3]) + epsilon_t)

        self.weights_input_hidden1 -= alpha_t * moment_weights[4] / (np.sqrt(variance_weights[4]) + epsilon_t)
        self.bias_input_hidden1 -= alpha_t * moment_biases[4] / (np.sqrt(variance_biases[4]) + epsilon_t)

    def mini_batch(self, epochs, batch_size, learning_rate, y, X):
        n = len(y)
        no_of_batches = int((n+batch_size-1)/batch_size)

        for i in range(epochs):
            for j in range(no_of_batches):
                X_train = X[j*batch_size : min((j+1)*batch_size, n)]
                y_train = y[j*batch_size : min((j+1)*batch_size, n)]
                out = self.forward_prop(X_train)
                self.back_prop(y_train, learning_rate, X_train)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required = True, help='root directory containing the 8 subfolders')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='root directory of test set')
    parser.add_argument('--save_weights_path', type=str, required = True, help='location to save the weights')
    
    parser.add_argument('--save_predictions_path', type=str, required=True, help='location to save predictions')
    args = parser.parse_args()
    #start = time.time()
    # Root directory containing the 8 subfolders
    root_dir_train = args.dataset_root #Path to the dataset directory
    root_dir_test = args.test_dataset_root
    weights_path = args.save_weights_path
    predic_path = args.save_predictions_path
    #mode = 'train' #Set mode to 'train' for loading the train set for training. Set mode to 'val' for testing your model after training. 

    

    # Create the custom dataset
      #Remember to import "numpy_transforms" functions.

    # Create the DataLoader
    train_dataset = TrainImageDataset(root_dir=root_dir_train, csv = os.path.join(root_dir_train, "train.csv"), transform=numpy_transform)
    train_dataloader = TrainDataLoader(train_dataset, batch_size=256)
    testing_dataset = TrainImageDataset(root_dir = root_dir_test, csv = os.path.join(root_dir_test, "val.csv"), transform = numpy_transform)
    testing_dataloader = TrainDataLoader(testing_dataset, batch_size=256)
    nn = NeuralNetwork(input_size=625, hidden_layer_1_size=512, hidden_layer_2_size=256, hidden_layer_3_size=128, hidden_layer_4_size= 32, output_size=8, batch_size=256)
    weights = [nn.weights_hidden4_output, nn.weights_hidden3_hidden4, nn.weights_hidden2_hidden3, nn.weights_hidden1_hidden2, nn.weights_input_hidden1]
    biases = [nn.bias_hidden4_output, nn.bias_hidden3_hidden4, nn.bias_hidden2_hidden3, nn.bias_hidden1_hidden2, nn.bias_input_hidden1]
    
    moment_weights = [np.zeros_like(weights[i]) for i in range(len(weights))]
    variance_weights = [np.zeros_like(weights[i]) for i in range(len(weights))]

    moment_biases = [np.zeros_like(biases[i]) for i in range(len(biases))]
    variance_biases = [np.zeros_like(biases[i]) for i in range(len(biases))]
    #print(variance_biases[1].shape)

    out = []
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-12
    alpha = 1e-3
    final_images = []
    final_labels = []
    final_test_images = []
    final_test_labels=[]
    n = 0
    n1=0

    x_labels = []
    train_loss = []
    test_loss = []


    for images, labels in train_dataloader:
        
        final_images.append(images)
        final_labels.append(np.eye(8)[labels])
        n += images.shape[0]

    for images, labels in testing_dataloader:
        final_test_images.append(images)
        final_test_labels.append(np.eye(8)[labels])
        n1+= images.shape[0]


    # Iterate through the DataLoader
    best_accuracy = 0

    for i in range(1500):
        
        match=0
        losses=0
        for j in range(len(final_images)):
              # one hot encoding
            out = nn.forward_prop(final_images[j])
            nn.back_prop(final_labels[j], 0.001, final_images[j], final_images[j].shape[0], alpha, beta1, beta2, moment_weights, variance_weights, moment_biases, variance_biases, epsilon, i)
            
            #predic = np.argmax(out, axis = 1)
            #print(predic.shape)
            #match += np.sum(np.argmax(final_labels[j], axis = 1) == predic)
            #tot += final_images[j].shape[0]
            #losses += loss(final_images[j], final_labels[j], 5, weights, biases)

        out = []
        
        closs=0
        match2=0
        for j in range(len(final_test_images)):

            # actual.extend(labels.tolist())
            
            predic = nn.forward_prop(final_test_images[j])
            predic = np.argmax(predic, axis=1)
            out.extend(predic.tolist())
            closs += loss(final_test_images[j], final_test_labels[j], len(weights), weights, biases)
            match2 += np.sum(np.argmax(final_test_labels[j], axis = 1)== predic)
        
        
        
        if(i==450):
            alpha = 1e-4
        
        #if(i%5==0 or i==999):
        #x_labels.append(i)
        #train_loss.append(-losses/n)
        #test_loss.append(-closs/n1)
        #print('loss testing is ', -closs/n1)
        #print('testing accuracy is ', match2/n1)
        #print(f"Loss after Epoch {i+1} is {-losses/n}")
        #print(f"Epoch {i + 1} completed in {time.time() - start} seconds")
        #print('accuracy is ', match/n)
        if(match2/n1 > best_accuracy):

            best_accuracy = match2/n1
            #print(best_accuracy)

            with open(predic_path + 'predictions.pkl', 'wb') as f :
                pickle.dump(out, f)
            
            network_params = {
                'weights': {},
                'bias': {}
            }

            network_params['weights']['fc1'] = nn.weights_input_hidden1
            network_params['weights']['fc2'] = nn.weights_hidden1_hidden2
            network_params['weights']['fc3'] = nn.weights_hidden2_hidden3
            network_params['weights']['fc4'] = nn.weights_hidden3_hidden4
            network_params['weights']['fc5'] = nn.weights_hidden4_output

            

            network_params['bias']['fc1'] = (nn.bias_input_hidden1.T).reshape(-1)
            network_params['bias']['fc2'] = (nn.bias_hidden1_hidden2.T).reshape(-1)
            network_params['bias']['fc3'] = (nn.bias_hidden2_hidden3.T).reshape(-1)
            network_params['bias']['fc4'] = (nn.bias_hidden3_hidden4.T).reshape(-1)
            network_params['bias']['fc5'] = (nn.bias_hidden4_output.T).reshape(-1)



            with open(weights_path + 'weights.pkl', 'wb') as f:
                pickle.dump(network_params, f)

            f.close()
    
    
    '''plt.plot(x_labels, train_loss, label="train_loss", color="blue")  # First line
    plt.plot(x_labels, test_loss, label="test_loss", color="red")   # Second line
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.show()
    out = []
    n1=0
    closs=0
    match2=0
    # actual = []
    for images, labels in testing_dataloader:
        n1+=images.shape[0]
        # actual.extend(labels.tolist())
        labels = np.eye(8)[labels]
        predic = nn.forward_prop(images)
        predic = np.argmax(predic, axis=1)
        out.extend(predic.tolist())
        closs += loss(images, labels, len(weights), weights, biases)
        match2 += np.sum(np.argmax(labels, axis = 1)== predic)
    
    print('loss testing is ', -closs/n1)
    print('accuracy testing is ', match2/n1)
    print(out)

    with open('predictions.pkl', 'wb') as f :
        pickle.dump(out, f)

    end = time.time()
    print('time taken : ', end - start)  

    f.close()  
        

    network_params = {
        'weights': {},
        'bias': {}
    }

    network_params['weights']['fc1'] = nn.weights_input_hidden1
    network_params['weights']['fc2'] = nn.weights_hidden1_hidden2
    network_params['weights']['fc3'] = nn.weights_hidden2_hidden3
    network_params['weights']['fc4'] = nn.weights_hidden3_hidden4
    network_params['weights']['fc5'] = nn.weights_hidden4_output

    

    network_params['bias']['fc1'] = (nn.bias_input_hidden1.T).reshape(-1)
    network_params['bias']['fc2'] = (nn.bias_hidden1_hidden2.T).reshape(-1)
    network_params['bias']['fc3'] = (nn.bias_hidden2_hidden3.T).reshape(-1)
    network_params['bias']['fc4'] = (nn.bias_hidden3_hidden4.T).reshape(-1)
    network_params['bias']['fc5'] = (nn.bias_hidden4_output.T).reshape(-1)



    with open('weights.pkl', 'wb') as f:
        pickle.dump(network_params, f)

    f.close()'''