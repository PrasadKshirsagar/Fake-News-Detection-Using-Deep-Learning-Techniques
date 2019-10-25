
import csv
with open('test.tsv', 'r') as inp, open('new_test.tsv', 'w') as out:
    writer = csv.writer(out, delimiter='\t')
    for row in csv.reader(inp, delimiter='\t'):
        if row[1] == "false" or row[1] == "true":
            writer.writerow(row)

with open('train.tsv', 'r') as inp, open('new_train.tsv', 'w') as out:
    writer = csv.writer(out, delimiter='\t')
    for row in csv.reader(inp, delimiter='\t'):
        if row[1] == "false" or row[1] == "true":
            writer.writerow(row)


with open('valid.tsv', 'r') as inp, open('new_valid.tsv', 'w') as out:
    writer = csv.writer(out, delimiter='\t')
    for row in csv.reader(inp, delimiter='\t'):
        if row[1] == "false" or row[1] == "true":
            writer.writerow(row)



import pandas as pd



w = pd.read_csv("new_train.tsv", usecols=[1,2], names=['label','text'],delimiter='\t')
w['label'] = w['label'].apply({True:0, False:1}.get)

w1 = pd.read_csv("new_test.tsv", usecols=[1,2], names=['label','text'],delimiter='\t')
w1['label'] = w1['label'].apply({True:0, False:1}.get)

w2 = pd.read_csv("new_valid.tsv", usecols=[1,2], names=['label','text'],delimiter='\t')
w2['label'] = w2['label'].apply({True:0, False:1}.get)

w.to_csv("liar_train.csv")
w1.to_csv("liar_test.csv")
w2.to_csv("liar_validation.csv")

print(w2)




