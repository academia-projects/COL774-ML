import pandas as pd
import numpy as np
import math
from argparse import ArgumentParser

# nltk.download('stopwords')
# nltk.download('punkt_tab')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


ps = PorterStemmer()
def remove_stopwords(text, stop_words):
    return [word for word in text if word not in stop_words]

def stem_text(text):
    return [ps.stem(word) for word in text]


def preprocess(texts, stop_words):
    texts = texts.apply(lambda row: row.lower().split()) # Extract the text data
    texts = texts.apply(lambda row: remove_stopwords(row, stop_words)) # Remove the stopwords
    texts = texts.apply(lambda row: stem_text(row)) # Apply stemming
    return texts

def generate_bigrams(tokens):
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

def predictions(X_test, priors, likelihoods, classes):
    log_l = np.log(likelihoods)
    log_r = np.log(1-likelihoods)
    log_p = np.log(priors)
    pred = X_test@log_l + (1-X_test)@log_r + log_p
    pred = np.argmax(pred, axis=1)
    y_pred = np.array([classes[i] for i in pred])
    return y_pred
 



parser = ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--stop", type=str, required=True)
args = parser.parse_args()

train_path = args.train
test_path = args.test
out_path = args.out
stop_path = args.stop

stop_words = set()

with open(stop_path, "r") as file:
    for line in file:
        stop_words.update(line.split())


data = pd.read_csv(train_path, sep='\t', header=None, quoting=3)
texts =  data.iloc[:, 2]
labels = data.iloc[:, 1]
#vocabulary = create_vocabulary(texts, stop_words)
texts = preprocess(texts, stop_words)
vocabulary = set()
for row in texts:
    vocabulary.update(row)
    bigram_tokens = generate_bigrams(row)
    vocabulary.update(bigram_tokens)

vocabulary = list(vocabulary)
vocab_size = len(vocabulary)

#print(vocabulary)

classes = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
X_train = np.zeros((len(texts), vocab_size + 1), dtype=np.float64)
y_train = np.zeros(len(labels), dtype = np.float64)
priors = np.zeros(len(classes), dtype=np.float64)
likelihoods = np.zeros((vocab_size + 1, len(classes)), dtype = np.float64)

word_index = {word: idx for idx, word in enumerate(vocabulary)}
class_index = {word: idx for idx, word in enumerate(classes)}

for index, label in enumerate(labels):
        y_train[index] = class_index[label]
    
for index, row in enumerate(texts):
    row_bigrams = generate_bigrams(row)
    temp = set(row)
    temp.update(row_bigrams)
    for word in temp:
        if (word in word_index) :
            X_train[index][word_index[word]] = 1
            likelihoods[word_index[word]][int(y_train[index])] += 1
        else :
            X_train[index][-1] = 1


for i in range(len(classes)):
    priors[i] = (y_train == i).sum()
    likelihoods[:, i] = (likelihoods[:, i] + 1) / (priors[i] + 2)
    priors[i] = priors[i]/len(y_train)

print(likelihoods) 

test_data = pd.read_csv(test_path, sep='\t', header=None, quoting=3)
test_texts =  test_data.iloc[:, 2]
test_labels = test_data.iloc[:, 1]
test_texts = preprocess(test_texts, stop_words)
X_test = np.zeros((len(test_texts), vocab_size + 1), dtype=np.float64)
y_test = np.zeros(len(test_labels), dtype = np.float64)

for index, row in enumerate(test_texts):
    row_bigrams = generate_bigrams(row)
    temp = set(row)
    temp.update(row_bigrams)
    for word in temp:
        if (word in word_index) :
            X_test[index][word_index[word]] = 1
        else :
            X_test[index][vocab_size] = 1

predicted_labels =predictions(X_train, priors, likelihoods, classes)

for i in range(len(classes)):
    print(f'Number of examples in class {classes[i]}: {(predicted_labels == classes[i]).sum()}')
    

# print(f'Accuracy is {(predicted_labels == test_labels).mean()}')

checker_data = np.load(r'checker_files\bernoulli_bigrams_probas_train.npy')
row_wise_argmax = np.argmax(checker_data, axis=1)
checker_labels = np.array([classes[i] for i in row_wise_argmax])

print(f'match_rate is {(predicted_labels == checker_labels).mean()}')
print(f'Incorrectly classified examples: {(predicted_labels != checker_labels).sum()}')


with open(out_path, "w") as file:
    for label in predicted_labels:
            file.write(f"{label}\n")



