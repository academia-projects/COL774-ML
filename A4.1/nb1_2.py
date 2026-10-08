import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.data import find
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle

def remove_stopwords(text, stop_words):
    return [word for word in text if word not in stop_words]

def stem_text(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]

def preprocess(x, stop_words):
    x = x.apply(lambda row: (row.lower()).split()) # Extract the text data
    x = x.apply(lambda row: remove_stopwords(row, stop_words)) # Remove the stopwords
    x = x.apply(stem_text) # Apply stemming
    return x



# Load the data
df = pd.read_csv('train.tsv', sep='\t', header=None, names = [f'Column_{i}' for i in range(14)], quoting = 3) # Load the data

# Preprocess the data
x = df['Column_2'] # Extract the text data
y = df['Column_1'] # Extract the labels

stop_words = set()
with open('stopwords.txt', 'r') as file:
    for line in file:
        stop_words.update(line.split())
x = preprocess(x, stop_words)

# construct vocabulary
vocabulary = set()
for row in x:
    vocabulary.update(row)

vocabulary = sorted(list(vocabulary))

# dictonary to map the index of every word to the word itself
word_index = {word: i for i, word in enumerate(vocabulary)}

classes =  ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
class_index = {word : i for i, word in enumerate(classes)}

print(f'Total number of unique words: {len(vocabulary)}')

# Create the numerical training data matrix
x_train = np.zeros((len(x), len(vocabulary)), dtype=np.float64)
y_train = np.zeros(len(x), dtype=int)

for i, row in enumerate(x):
    for word in row:
        x_train[i, word_index[word]] += 1
    
for i, label in enumerate(y):
    y_train[i] = class_index[label]

# create the weight matrix for the classes which is basically the probabilities of every word belonging to a specific class
w = np.zeros((len(vocabulary) + 1, len(classes)), dtype=np.float64)

# train the model OR calculate the probabilities of every word belonging to a specific class

# calculate the occurence of every word in every class
for i, words in enumerate(x):
    for j in words:
        w[word_index[j], y_train[i]] += 1

# find the probabilities of each class
p_class = np.zeros(len(classes), dtype=np.float64)
for i in range(len(classes)):
    p_class[i] = (y_train == i).sum() / len(y_train)
print(f'Probabilities of each class: {p_class}')

# calculate the probabilities 
# Divide by the total number of words in each class - MULTINOMIAL DIFFERENCE FROM BERNOULLI
w = (w + 1) / (w.sum(axis=0, keepdims=True) + len(vocabulary))

print('-'*50)

# load the validation data
df = pd.read_csv('valid.tsv', sep='\t', header=None, names = [f'Column_{i}' for i in range(14)], quoting = 3) # Load the data
x = df['Column_2'] # Extract the text data
y = df['Column_1'] # Extract the labels

x = preprocess(x, stop_words)

x_valid = np.zeros((len(x), len(vocabulary) + 1), dtype=np.float64)
y_valid = np.zeros(len(x), dtype=int)

for i, row in enumerate(x):
    for word in row:
        if word in word_index:
            x_valid[i, word_index[word]] += 1       # changed from 1 to count of word in the text - MULTINOMIAL DIFFERENCE FROM BERNOULLI
        else :
            x_valid[i, -1] += 1

for i, label in enumerate(y):
    y_valid[i] = class_index[label]

# predict the classes
y_pred_m = (x_valid @ np.log(w)) + np.log(p_class)       # probability of being absent is removed as it is not needed in multinomial
# print(np.exp(y_pred))
y_pred = np.argmax(y_pred_m, axis=1)

# calculate the number of examples in each predicted class
for i in range(len(classes)):
    print(f'Number of examples in class {classes[i]}: {(y_pred == i).sum()}')

# calculate the accuracy
accuracy = (y_pred == y_valid)
print(f'Accuracy: {accuracy.mean()}')

checker_data = np.load(r'checker_files\multinomial_probas_test.npy')
# # print(checker_data)

row_wise_argmax = np.argmax(checker_data, axis=1)
checker_labels = np.array([class_index[classes[i]] for i in row_wise_argmax])
# print(checker_labels.__len__(), "   " , y_pred.__len__())
print(f'match_rate is {(y_pred == checker_labels).mean()}')
print(f'Incorrectly classified examples: {(y_pred != checker_labels).sum()}')
# print(np.exp(y_pred_m[y_pred != checker_labels]))
# print('-'*50)
# print(checker_data[checker_labels != y_pred])