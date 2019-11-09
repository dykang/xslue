import os
import glob
import pandas as pd
import numpy as np

dataset = 'DailyDialog'

dts = ['train','validation', 'test']
dts_out = [ 'train', 'dev', 'test']

# data = pd.read_csv(os.path.join('../slue_data/', dataset, 'original','emobank.csv'),index_col=0)

# # random split
# msk = np.random.rand(len(data)) < 0.9
# train = data[msk]
# devtest = data[~msk]
# msk = np.random.rand(len(devtest)) < 0.5
# dev = devtest[msk]
# test = devtest[~msk]
# print(len(train), len(dev), len(test))

# # write to files
# for dt_out,d in zip(dts_out, [train, dev, test]):
    # fout = open(os.path.join('../slue_data',dataset,'{}.tsv'.format(dt_out)),'w')
    # for index, row in d.iterrows():
        # fout.write('{}\t{}\t{}\t{}\t{}\n'.format(index,row.V, row.A, row.D, row.text))
    # fout.close()


for dt,dt_out in zip(dts,dts_out):
    fout = open(os.path.join('../slue_data',dataset,'{}.tsv'.format(dt_out)),'w')
    dial = open(os.path.join('../slue_data/', dataset, 'processed', dt, 'dial.txt')).readlines()
    emo = open(os.path.join('../slue_data/', dataset, 'processed', dt, 'emo.txt')).readlines()

    assert len(dial) == len(emo)
    for d,e in zip(dial,emo):
        fout.write('{}\t{}\n'.format(d.strip(),e.strip()))
    fout.close()

