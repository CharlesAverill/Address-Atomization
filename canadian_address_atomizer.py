from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from random_words import RandomWords

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

def word2features(sent, i):
    word = str(sent[i][0])
    postag = str(sent[i][1])

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = str(sent[i-1][0])
        postag1 = str(sent[i-1][1])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = str(sent[i+1][0])
        postag1 = str(sent[i+1][1])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                     s["tags"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        s = self.grouped["Sentence: {}".format(self.n_sent)]
        self.n_sent += 1
        return s

def generate_random_string_from_list(ls):
    out = ""
    while len(ls) > 0:
        out += ls.pop(random.randint(0, len(ls) - 1)).strip() + " "
    return out

def generate_postal_code():
    _ = [2, 3, 4, 5, 6, 7]
    out = ""
    abcs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = "0123456789"
    for i in _:
        if(i % 2 == 0):
            x = random.randint(0, len(abcs))
            out += abcs[x:x+1]
        if(i % 2 != 0):
            x = random.randint(0, len(nums))
            out += nums[x:x+1]
    return out

#data loading

X = []
y = []

#xl = pd.ExcelFile(input("Please enter filename: "))
#df = xl.parse("CANADA").fillna("")

sent_nums = []
words = []
tags = []

#number_first_prefix 0
#number_first 1
#number_first_suffix 2
#number_last_prefix 3
#number_last 4
#number_last_suffix 5
#street_direction_abbreviation 6
#street_name 7
#street_type_abbreviation 8
#unit_type 9
#unit_number_prefix 10
#unit_number 11
#unit_number_suffix 12
#postal_code 13
#city 14

count = 0

r = RandomWords()
abbreviations = ["Ave.", "Blvd.", "Bldg.", "Crt.", "Cres.", "Dr.", "Pl.", "Rd.", "Sq.", "Stn.", "St.", "Terr."]
cities = ['Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Edmonton', 'Ottawa-Gatineau', 'Winnipeg', 'Quebec City', 'Hamilton', 'Kitchener', 'London', 'Victoria', 'Halifax', 'Oshawa', 'Windsor']

for i in range(2500):
    if i % 1000 == 0:
        print(i)
    words.append(random.randint(10000, 99999))
    tags.append(1)
    sent_nums.append("Sentence: " + str(count))


    dirs = ['N', 'S', 'E', 'W']
    words.append(random.choice(dirs))
    tags.append(6)
    sent_nums.append("Sentence: " + str(count))


    words.append(r.random_word() + " " + r.random_word())
    tags.append(7)
    sent_nums.append("Sentence: " + str(count))

    words.append(generate_postal_code())
    tags.append(13)
    sent_nums.append("Sentence: " + str(count))

    words.append(random.choice(cities))
    tags.append(14)
    sent_nums.append("Sentence: " + str(count))

    count += 1

"""
for ind, row in df.iterrows():
    if(row['number_first_prefix']):
        words.append(row['number_first_prefix'])
        tags.append(0)
        sent_nums.append("Sentence: " + str(count))
    if(row['number_first']):
        words.append(random.randint(10000, 99999))
        tags.append(1)
        sent_nums.append("Sentence: " + str(count))
    if(row['number_first_suffix']):
        words.append(row['number_first_suffix'])
        tags.append(2)
        sent_nums.append("Sentence: " + str(count))
    if(row['number_last_prefix']):
        words.append(row['number_last_prefix'])
        tags.append(3)
        sent_nums.append("Sentence: " + str(count))
    if(row['number_last']):
        words.append(row['number_last'])
        tags.append(4)
        sent_nums.append("Sentence: " + str(count))
    if(row['number_last_suffix']):
        words.append(row['number_last_suffix'])
        tags.append(5)
        sent_nums.append("Sentence: " + str(count))
    if(row['street_direction_abbreviation']):
        dirs = ['N', 'S', 'E', 'W']
        words.append(random.choice(dirs))
        tags.append(6)
        sent_nums.append("Sentence: " + str(count))
    if(row['street_name']):
        words.append(r.get_random_word() + " " + r.get_random_word())
        tags.append(7)
        sent_nums.append("Sentence: " + str(count))
    if(row['street_type_abbreviation']):
        words.append(random.choice(abbreviation))
        tags.append(8)
        sent_nums.append("Sentence: " + str(count))
    if(row['unit_type']):
        words.append(row['unit_type'])
        tags.append(9)
        sent_nums.append("Sentence: " + str(count))
    if(row['unit_number_prefix']):
        words.append(row['unit_number_prefix'])
        tags.append(10)
        sent_nums.append("Sentence: " + str(count))
    if(row['unit_number']):
        words.append(row['unit_number'])
        tags.append(11)
        sent_nums.append("Sentence: " + str(count))
    if(row['unit_number_suffix']):
        words.append(row['unit_number_suffix'])
        tags.append(12)
        sent_nums.append("Sentence: " + str(count))

    #Postal code
    words.append(generate_postal_code())
    tags.append(13)
    sent_nums.append("Sentence: " + str(count))

    if(row['city']):
        words.append(random.choice(cities))
        tags.append(14)
        sent_nums.append("Sentence: " + str(count))
    count += 1
"""
for i in range(len(words)):
    word = words[i]
    if isinstance(word, float):
        words[i] = int(word)
    words[i] = str(words[i])

tags = [n + 1 for n in tags]

words.append("ENDPAD")
n_words = len(words)
n_tags = len(tags)

df = pd.DataFrame(list(zip(sent_nums, words, tags)), columns=["Sentence #", "words", "tags"])

getter = SentenceGetter(df)
sentences = getter.sentences

max_len = 20
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
tag2idx["0"] = 0

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["0"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

model = Model(input, out)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=3, validation_split=0.1, verbose=1)

hist = pd.DataFrame(history.history)
plt.figure(figsize=(12,12))
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.show()

i = random.randint(0, len(X_test))
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(X_test[i], p[0]):
    print("{:15}: {}".format(words[w], tags[pred]))

model.save_weights("atomizer.h5")
