
import pandas as pd
import numpy as np


df = pd.read_csv("train.csv")

train_data = df.sample(frac=1).reset_index(drop=True)

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

train_df.to_csv("kaggle_train.csv")
test_df.to_csv("kaggle_test.csv")
validation_df.to_csv("kaggle_validation.csv")



