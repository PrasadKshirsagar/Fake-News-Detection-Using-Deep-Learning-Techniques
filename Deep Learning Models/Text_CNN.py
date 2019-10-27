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



# Defining parameters
stops = set(stopwords.words('english'))
vocab_size = 10000
sequence_length = 200
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
#train_data['text'] = train_data['text'].str.replace("[^0-9a-zA-Z]", " ")
data = train_data["text"]
labels = np.array(train_data["label"])

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
print(X_test.shape ,y_test.shape)
print(X_val.shape ,y_val.shape)



num_filters=2
dropOut=0.50

# Creating our text-CNN model :
input = Input(shape=(sequence_length, ))
aft_emb = Embedding(vocab_size, embedding_dim)(input)
aft_emb = Dropout(dropOut)(aft_emb)

# Function to create convolution models
def models(kernal_size):
    x = Conv1D(nb_filter = num_filters, kernel_size = kernal_size, border_mode = 'valid', activation = 'relu',strides = 1)(aft_emb)
    x_out = GlobalMaxPooling1D()(x)
    return x_out




# Model1 of filters having kernel size as 2
x_out = models(2)

# Model2 of filters having kernel size as 3
y_out = models(3)

# Model3 of filters having kernel size as 4
z_out = models(4)

# Concatenate the outputs from above 3 models
concatenated = concatenate([x_out, y_out, z_out])

# Apply dense layers 
dense1 = Dense(250)(concatenated)
dense1 = LeakyReLU(alpha=0.05)(dense1)
out = Dense(2, activation = 'softmax', name = 'output_layer')(dense1)

# Get final model
merged_model = Model(input, out)

# Creating Tensorboard
#tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(merged_model.summary())




# Model fitting
history = merged_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128)





# Preprocessing Test data

# Checking on liar Dataset 

#test_data = pd.read_csv("../input/btp-dataset/test_kaggle.csv")
#test_data['text'] = test_data['text'].str.replace("[^0-9a-zA-Z]", " ")

test_data = pd.read_csv("../input/liar-dataset/new_train.tsv", usecols=[1,2], names=['label','text'],delimiter='\t')
test_data['label'] = test_data['label'].replace(regex='false', value=1)
test_data['label'] = test_data['label'].replace(regex='true', value=0)



test_x = test_data["text"]
labels = np.array(test_data["label"])

text = []
for sentence in test_x:
    text.append(" ".join([word for word in str(sentence).split() if word not in stops]))

X_test = preprocess(text)
#print(X_test)
y_test = np_utils.to_categorical(labels , num_classes=num_of_classes) 



# Final evaluation of the model
scores = merged_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Confusion Matrix
y_pred = merged_model.predict(X_test)
y_pred = [ np.argmax(t) for t in y_pred ]
y_test_non_category = [ np.argmax(t) for t in y_test]
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test_non_category, y_pred)
print(conf_mat)



