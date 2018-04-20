filename = 'text_scores.csv'

import json
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer

def load_df(filename):
    raw = pd.read_csv(filename, index_col='index')
    df = pd.DataFrame(raw)
    X_train, y_train = df['text'], df['score']
    training = [(x,y) for x in X_train for y in y_train]
    return training

# load the text and score as a list of tuples
training = load_df(filename)

# create training data from the text
train_x = [x[0] for x in training]
# index all the sentiment labels
train_y = np.asarray([x[1] for x in training])

# only work with the 5000 most popular words found in our dataset
max_words = 5000

# create a new Tokenizer and feed texts to the Tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

def convert_text_to_index_array(text):
    # use `text_to_word_sequence` to make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
# for each text, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all texts converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed texts
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = keras.utils.to_categorical(train_y, 2)

# Building the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=128,
          epochs=3,
          validation_split=0.1,
          shuffle=True)

# Final evaluation of the model
scores = model.evaluate(train_x, train_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
