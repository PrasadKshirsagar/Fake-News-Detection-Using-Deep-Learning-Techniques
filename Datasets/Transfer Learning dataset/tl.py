import pandas as pd


f=pd.read_csv("kaggle_test.csv")
keep_col = ['title','label']
new_f = f[keep_col]
new_f.columns = ['text', 'label']
df_reorder = new_f[['label','text']] 
df_reorder.to_csv("k_test.csv", index=False)

f1=pd.read_csv("kaggle_validation.csv")
keep_col1 = ['title','label']
new_f1 = f1[keep_col1]
new_f1.columns = ['text', 'label']
df_reorder = new_f1[['label','text']] 
df_reorder.to_csv("k_val.csv", index=False)

f2=pd.read_csv("isot_validation.csv")
keep_col2 = ['title','labels']
new_f2 = f2[keep_col2]
new_f2.columns = ['text', 'label']
df_reorder = new_f2[['label','text']] 
df_reorder.to_csv("i_val.csv", index=False)

f3=pd.read_csv("isot_test.csv")
keep_col3 = ['title','labels']
new_f3 = f3[keep_col3]
new_f3.columns = ['text', 'label']
df_reorder = new_f3[['label','text']] 
df_reorder.to_csv("i_test.csv", index=False)


f2=pd.read_csv("liar_validation.csv")
keep_col3 = ['label','text']
new_f3 = f2[keep_col3]
new_f3.columns = ['label','text']
df_reorder1 = new_f3[['label','text']] 
df_reorder1.to_csv("l_val.csv", index=False)

f2=pd.read_csv("liar_test.csv")
keep_col3 = ['label','text']
new_f3 = f2[keep_col3]
new_f3.columns = ['label','text']
df_reorder2 = new_f3[['label','text']] 
df_reorder2.to_csv("l_test.csv", index=False)

f2=pd.read_csv("liar_train.csv")
keep_col3 = ['label','text']
new_f3 = f2[keep_col3]
new_f3.columns = ['label','text']
df_reorder3 = new_f3[['label','text']] 
df_reorder3.to_csv("l_train.csv", index=False)

pdList = [df_reorder1, df_reorder2, df_reorder3]  # List of your dataframes
new_df = pd.concat(pdList)
new_df.to_csv("l_total.csv", index=False)
print(new_df.shape)