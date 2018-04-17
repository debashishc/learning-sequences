
# import csv

filename = 'text_scores.csv'
# def read_csv(filename):
#     text_score = dict()
#     with open(filename, 'r') as f:
#         data = csv.reader(f)
#         next(data)
#         for row in data:
#             idx, text, score = row[0], row[1], row[2]
#             text_score[idx, text] = score 
#     return text_score

import pandas as pd 
# from sklearn.model_selection import train_test_split

def load_df(filename):
    raw = pd.read_csv(filename, index_col='index')
    df = pd.DataFrame(raw)
    X_train, y_train = df['text'], df['score']
    training = [(x,y) for x in X_train for y in y_train]
    return training


training = load_df(filename)

# # # LSTM with Dropout for sequence classification in the IMDB dataset
# import numpy
# from keras.datasets import imdb
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence

# # # load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# # print(X_train[1:10])
# # truncate and pad input sequences
# max_review_length = 500
# print(X_train)
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# # create the model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(Dropout(0.2))
# model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, epochs=3, batch_size=64)

# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

import numpy as np

# create our training data from the tweets
train_x = [x[0] for x in training]
# index all the sentiment labels
train_y = np.asarray([x[1] for x in training])

import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer

# only work with the 3000 most popular words found in our dataset
max_words = 5000

# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit(train_x, train_y,
#           batch_size=64,
#           epochs=5,
#           verbose=1,
#           validation_split=0.1,
#           shuffle=True)
          
model.fit(train_x, train_y, epochs=3, batch_size=128, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(train_x, train_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
