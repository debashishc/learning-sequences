
import csv

filename = 'text_scores.csv'
def read_csv(filename):
    text_score = dict()
    with open(filename, 'r') as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            idx, text, score = row[0], row[1], row[2]
            text_score[idx, text] = score 
    return text_score

import pandas as pd 
from sklearn.model_selection import train_test_split

def load_df(filename):
    columns = ['index','text','score']
    raw = pd.read_csv(filename, index_col='index')
    df = pd.DataFrame(raw)
    print(df)

load_df(filename)


# # LSTM with Dropout for sequence classification in the IMDB dataset
# import numpy
# from keras.datasets import imdb
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence

# # load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# print(X_train[1:10])
# # truncate and pad input sequences
# max_review_length = 500
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
