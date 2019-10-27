import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense,GlobalMaxPooling1D
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dropout,add
import numpy as np
from keras.layers import Input
import pandas as pd
from keras.layers import LeakyReLU
from nltk.corpus import stopwords
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Dropout, Embedding, Bidirectional,CuDNNLSTM
from keras.layers import Dense,LSTM



fake = [1]*23481
true = [0]*21417

import pandas

df_true = pandas.read_csv("../input/btp-dataset/True.csv")

df_true['labels'] = true
df_true.to_csv('new_true.csv')

df_fake = pandas.read_csv("../input/btp-dataset/Fake.csv")

df_fake['labels'] = fake
df_fake.to_csv('new_fake.csv')

train_data = pd.concat([df_true,df_fake])



# Defining parameters
stops = set(stopwords.words('english'))
vocab_size = 10000
sequence_length = 25
embedding_dim = 50
num_of_classes = 2



# Preprocessing of the data
def preprocess(text):
    sequences = tokenizer.texts_to_sequences(text)
    X = pad_sequences(sequences, padding='pre', maxlen=sequence_length)
    X = np.array(X)
    return X



# Get sentences and labels from file
#train_data = pd.read_csv("../input/dataset/data_combined_train.csv")
data = train_data["title"]
labels = np.array(train_data["labels"])

texts = []
for itr in data:
    texts.append(" ".join([word for word in str(itr).split() if word not in stops]))



# Using tokenizer API for tokenization
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
X_texts = preprocess(texts)
y_labels = np_utils.to_categorical(labels , num_classes=num_of_classes)



# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_texts, y_labels, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)
print(X_train.shape ,y_train.shape)
print(X_val.shape ,y_val.shape)



"""Building the Model"""

dim = 100
dropout = 0.2
lstm_out = 400

def Build_Model(vocab_sizze, dim, dropout, lstm_out):
    """Defining the model"""
    #tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)
    model = Sequential()
    e = Embedding(input_dim = vocab_size, output_dim = dim, input_length = X_train.shape[1], dropout = dropout) 
    model.add(e)
    model.add(Bidirectional(CuDNNLSTM(lstm_out, return_sequences = True)))
    model.add(Dropout(dropout))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model

model = Build_Model(vocab_size, dim, dropout, lstm_out)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())




# Model fitting
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=4, batch_size=128)



# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


