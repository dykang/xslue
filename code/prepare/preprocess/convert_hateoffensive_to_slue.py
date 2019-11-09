import os
import glob
import pandas as pd
import numpy as np
import random
import pickle
import sys
import nltk
dataset = 'HateOffensive'

# dts = ['train','dev', 'test']
dts_out = [ 'train', 'dev', 'test']
data_slue = '/data/slue'

data = pd.read_csv("/data/slue/HateOffensive/original/labeled_data.csv")

data = np.array(data)

# 0 - hate speech
# 1 - offensive  language
# 2 - neither
# random split
random.shuffle(data)
msk = np.random.rand(len(data)) < 0.9
train = data[msk]
devtest = data[~msk]
msk = np.random.rand(len(devtest)) < 0.5
dev = devtest[msk]
test = devtest[~msk]
print(len(train), len(dev), len(test))


# write to files
for d,dt_out in zip([train,dev,test], dts_out):
    print('Reading...',dt_out)
    # data = pd.read_csv(os.path.join(data_slue, dataset, 'preprocessed','sarc','{}.txt'.format(dt)),index_col=0,header=None,sep='\t', lineterminator='\n')

    fout = open(os.path.join(data_slue,dataset,'{}.tsv'.format(dt_out)),'w')
    for row in d:
        text = row[6].replace('\r',' ').replace('\n',' ').strip()
        fout.write('{}\t{}\t{}\n'.format(row[0], row[5], text))
    fout.close()


