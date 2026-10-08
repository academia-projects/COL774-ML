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
x_train = np.zeros((len(x), len(vocabulary)), dtype=int)
y_train = np.zeros(len(x), dtype=int)

for i, row in enumerate(x):
    for word in row:
        x_train[i, word_index[word]] = 1
    
for i, label in enumerate(y):
    y_train[i] = class_index[label]

# create the weight matrix for the classes which is basically the probabilities of every word belonging to a specific class
w = np.zeros((len(vocabulary) + 1, len(classes)), dtype=np.float64)

# train the model OR calculate the probabilities of every word belonging to a specific class

# calculate the occurence of every word in every class
for i, words in enumerate(x):
    temp = set(words)
    for j in temp:
        w[word_index[j], y_train[i]] += 1

# find the probabilities of each class
p_class = np.zeros(len(classes), dtype=np.float64)
for i in range(len(classes)):
    p_class[i] = (y_train == i).sum() / len(y_train)
print(f'Probabilities of each class: {p_class}')

# calculate the probabilities
w = (w + 1)
for i in range(6):
    w[:, i] = w[: , i]/((y_train == i).sum()+2) # doubt in the denominator of laplace smoothing

# taking log so the multiplication becomes addition and the values don't become too small
w_pres = np.log(w)
w_abs = np.log(1 - w)

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
            x_valid[i, word_index[word]] = 1
        else :
            x_valid[i, -1] = 1

for i, label in enumerate(y):
    y_valid[i] = class_index[label]

# predict the classes
y_pred = (x_valid @ w_pres) + ((1 - x_valid) @ w_abs) + np.log(p_class)
y_pred = np.argmax(y_pred, axis=1)

print(y_pred)

# calculate the number of examples in each predicted class
for i in range(len(classes)):
    print(f'Number of examples in class {classes[i]}: {(y_pred == i).sum()}')

# calculate the accuracy
accuracy = (y_pred == y_valid).mean()
print(f'Accuracy: {accuracy}')

# with open('word_count.pkl', 'rb') as file:
#     checker_data = pickle.load(file)
# print(len(checker_data['pants-fire']))
# #check if the w values are correct
# for i in range(6):
#     temp = np.array(list(checker_data[classes[i]]), dtype=np.float64)
#     temp -= w[: , i]
#     temp = np.where(temp > 0, 1, 0)
#     # temp = np.sort(temp)
#     # print(temp[-10:])
#     # print(temp[:10])
#     print(np.sum(temp))
#     # print(np.allclose(w[:, i], temp))

checker_data = np.load(r'checker_files\bernoulli_probas_test.npy')
row_wise_argmax = np.argmax(checker_data, axis=1)
checker_labels = np.array([class_index[classes[i]] for i in row_wise_argmax])
# print(checker_labels.shape, ' ' , y_pred.shape)
print(f'match_rate is {(y_pred == checker_labels).mean()}')
print(f'Incorrectly classified examples: {(y_pred != checker_labels).sum()}')