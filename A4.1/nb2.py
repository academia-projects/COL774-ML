import pandas as pd
import numpy as np
import math
from argparse import ArgumentParser
import nltk

# nltk.download('stopwords')
# nltk.download('punkt_tab')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


ps = PorterStemmer()
def remove_stopwords(text, stop_words):
    return [word for word in text if word not in stop_words and word.isalnum()]

def stem_text(text):
    return [ps.stem(word) for word in text]


def preprocess(texts, stop_words):
    texts = texts.apply(lambda row: nltk.word_tokenize(row)) # Extract the text data
    texts = texts.apply(lambda row: remove_stopwords(row, stop_words)) # Remove the stopwords
    texts = texts.apply(lambda row: stem_text(row)) # Apply stemming
    return texts

def generate_bigrams(tokens):
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)] + [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]

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
speakers = data.iloc[:, 4]
subjects = data.iloc[:, 3]
party_aff = data.iloc[:, 7]
contexts = data.iloc[:, 13]
texts = texts.fillna('').astype(str)
speakers = speakers.fillna('').astype(str)
subjects = subjects.fillna('').astype(str)
party_aff = party_aff.fillna('').astype(str)
contexts = contexts.fillna('').astype(str)
labels = data.iloc[:, 1]
#vocabulary = create_vocabulary(texts, stop_words)
texts = preprocess(texts, stop_words)
subjects = preprocess(subjects, stop_words)
contexts = preprocess(contexts, stop_words)
vocabulary = set()
for row in texts:
    vocabulary.update(row)
    bigram_tokens = generate_bigrams(row)
    vocabulary.update(bigram_tokens)

for speaker in speakers:
    vocabulary.update(speaker)

for sub in subjects:
    vocabulary.update(sub)
    bigram_tokens = generate_bigrams(sub)
    vocabulary.update(bigram_tokens)

for party in party_aff:
    vocabulary.update(party)

for context in contexts:
    vocabulary.update(context)
    bigram_tokens = generate_bigrams(context)
    vocabulary.update(bigram_tokens)

vocabulary = list(vocabulary)
vocab_size = len(vocabulary)

#print(vocabulary)
print(f'Total number of unique words: {vocab_size}')

classes = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
X_train = np.zeros((len(texts), vocab_size), dtype=np.float64)
y_train = np.zeros(len(labels), dtype = np.float64)
priors = np.zeros(len(classes), dtype=np.float64)
likelihoods = np.zeros((vocab_size + 1, len(classes)), dtype = np.float64)

word_index = {word: idx for idx, word in enumerate(vocabulary)}
class_index = {word: idx for idx, word in enumerate(classes)}

for index, label in enumerate(labels):
        y_train[index] = class_index[label]
    
for index, row in enumerate(texts):
    row_bigrams = generate_bigrams(row)
    total_tokens = row_bigrams+row
    for word in total_tokens:
        if (word in word_index) :
            X_train[index][word_index[word]] += 1
            likelihoods[word_index[word]][int(y_train[index])] += 1

for index, speaker in enumerate(speakers):
    if (speaker in word_index) :
        X_train[index][word_index[speaker]] += 1
        likelihoods[word_index[speaker]][int(y_train[index])] += 1

for index, sub in enumerate(subjects):
    row_bigrams = generate_bigrams(sub)
    total_tokens = row_bigrams+sub
    for s in total_tokens:
        if(s in word_index):
            X_train[index][word_index[s]] += 1
            likelihoods[word_index[s]][int(y_train[index])] += 1

for index, party in enumerate(party_aff):
    if(party in word_index):
        X_train[index][word_index[party]] += 1
        likelihoods[word_index[party]][int(y_train[index])] += 1

for index, context in enumerate(contexts):
    row_bigrams = generate_bigrams(context)
    total_tokens = row_bigrams+context
    for con in total_tokens:
        if(con in word_index):
            X_train[index][word_index[con]] += 1
            likelihoods[word_index[con]][int(y_train[index])] += 1


for i in range(len(classes)):
    priors[i] = (y_train == i).sum() / len(y_train)
    
likelihoods = (likelihoods + 1) / (likelihoods.sum(axis=0, keepdims=True) + vocab_size)

test_data = pd.read_csv(test_path, sep='\t', header=None, quoting=3)
test_texts =  test_data.iloc[:, 2]
test_speakers = test_data.iloc[:, 4]
test_subjects = test_data.iloc[:, 3]
test_party_aff = test_data.iloc[:, 7]
test_contexts = test_data.iloc[:, 13]
test_labels = test_data.iloc[:, 1]
test_texts = test_texts.fillna('').astype(str)
test_speakers = test_speakers.fillna('').astype(str)
test_subjects = test_subjects.fillna('').astype(str)
test_party_aff = test_party_aff.fillna('').astype(str)
test_contexts = test_contexts.fillna('').astype(str)
test_texts = preprocess(test_texts, stop_words)
test_subjects = preprocess(test_subjects, stop_words)
test_contexts = preprocess(test_contexts, stop_words)
X_test = np.zeros((len(test_texts), vocab_size + 1), dtype=np.float64)
y_test = np.zeros(len(test_labels), dtype = np.float64)

for index, row in enumerate(test_texts):
    row_bigrams = generate_bigrams(row)
    total_tokens = row_bigrams+row
    for word in total_tokens:
        if (word in word_index) :
            X_test[index][word_index[word]] += 1
        else :
            X_test[index][vocab_size] += 1

for index, speaker in enumerate(test_speakers):
    if (speaker in word_index) :
        X_test[index][word_index[speaker]] += 1
    else:
        X_test[index][vocab_size] += 1

for index, sub in enumerate(test_subjects):
    row_bigrams = generate_bigrams(sub)
    total_tokens = row_bigrams+sub
    for s in total_tokens:
        if(s in word_index):
            X_test[index][word_index[s]] += 1
        else:
            X_test[index][vocab_size] += 1

for index, party in enumerate(test_party_aff):
    if(party in word_index):
        X_test[index][word_index[party]] += 1
    else:
        X_test[index][vocab_size] += 1

for index, context in enumerate(test_contexts):
    row_bigrams = generate_bigrams(context)
    total_tokens = row_bigrams+context
    for con in total_tokens:
        if(con in word_index):
            X_test[index][word_index[con]] += 1
        else:
            X_test[index][vocab_size] += 1

predicted_labels =predictions(X_test, priors, likelihoods, classes)

for i in range(len(classes)):
    print(f'Number of examples in class {classes[i]}: {(predicted_labels == classes[i]).sum()}')
    

print(f'Accuracy is {(predicted_labels == test_labels).mean()}')

# checker_data = np.load(r'checker_files\multinomial_bigrams_probas_test.npy')
# row_wise_argmax = np.argmax(checker_data, axis=1)
# checker_labels = np.array([classes[i] for i in row_wise_argmax])

# print(f'match_rate is {(predicted_labels == checker_labels).mean()}')

with open(out_path, "w") as file:
    for label in predicted_labels:
            file.write(f"{label}\n")