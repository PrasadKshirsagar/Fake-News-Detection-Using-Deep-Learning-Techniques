# import libraries
import fastai
from fastai import *
from fastai.text import * 
import pandas as pd
import numpy as np
from functools import partial
import io
import os



from sklearn.model_selection import train_test_split

# create a dataframe
#df = pd.read_csv('l_val.csv')
#df_test = pd.read_csv('k_val.csv')

df1 = pd.read_csv('l_train.csv')
df2 = pd.read_csv('k_test.csv')
df3 = pd.read_csv('i_test.csv')

df1, t5 = train_test_split(df1, test_size = 0.85, random_state = 10)
df2, t3 = train_test_split(df2, test_size = 0.87, random_state = 10)
df3, t4 = train_test_split(df3, test_size = 0.94, random_state = 10)

pdList = [df1, df2, df3]  # List of your dataframes
df = pd.concat(pdList,ignore_index=True)

df_test = pd.read_csv('l_val.csv')




df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
df = df.fillna(" ")

df_test['text'] = df_test['text'].str.replace("[^a-zA-Z]", " ")
df_test = df_test.fillna(" ")




import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
stop_words = stopwords.words('english') 




# tokenization on validation data
tokenized_doc = df['text'].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words]) 

# de-tokenization 
detokenized_doc = [] 
for i in range(len(df)): 
    t = ' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 
df['text'] = detokenized_doc




# tokenization on test data  
tokenized_doc1 = df_test['text'].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc1 = tokenized_doc1.apply(lambda x: [item for item in x if item not in stop_words]) 

# de-tokenization 
detokenized_doc1 = [] 
for i in range(len(df_test)): 
    t1 = ' '.join(tokenized_doc1[i]) 
    detokenized_doc1.append(t1) 
df_test['text'] = detokenized_doc1



# split data into training and validation set

df_trn = df
df_val = df_test


# Language model data
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")


# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


learn = language_model_learner(data_lm,  arch = AWD_LSTM, pretrained = True, drop_mult=0.1)


# train the learner object
learn.fit_one_cycle(7, 1e-2)


learn.lr_find()


learn.recorder.plot()


learn.save_encoder('ft_enc')


learn = text_classifier_learner(data_clas,AWD_LSTM, drop_mult=0.1)
learn.load_encoder('ft_enc')


learn.lr_find()
learn.recorder.plot()


learn.fit_one_cycle(6, 1e-2)


# get predictions
preds, targets = learn.get_preds()

predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)









