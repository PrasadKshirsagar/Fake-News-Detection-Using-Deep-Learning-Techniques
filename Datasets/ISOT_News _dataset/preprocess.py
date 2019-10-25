fake = [1]*23481
true = [0]*21417

import pandas as pd
import pandas
import numpy as np


df_true = pandas.read_csv("True.csv")

df_true['labels'] = true
df_true.to_csv('new_true.csv')

df_fake = pandas.read_csv("Fake.csv")

df_fake['labels'] = fake
df_fake.to_csv('new_fake.csv')

train_data = pd.concat([df_true,df_fake],ignore_index=True)

# Shuffling dataframe
train_data = train_data.sample(frac=1).reset_index(drop=True)

#partitioning into train,test and validation

msk = np.random.rand(len(train_data)) < 0.8

train_temp = train_data[msk]

val = np.random.rand(len(train_temp)) < 0.0625

train_df = train_temp[~val]
validation_df = train_temp[val]
test_df = train_data[~msk]

print(len(train_df))
print(len(test_df))
print(len(validation_df))

train_df.to_csv("isot_train.csv")
test_df.to_csv("isot_test.csv")
validation_df.to_csv("isot_validation.csv")










