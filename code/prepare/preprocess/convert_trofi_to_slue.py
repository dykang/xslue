import os
import glob
import pandas as pd
import numpy as np

dataset = 'TroFi'

dts = ['train','val', 'test']
dts_out = [ 'train', 'dev', 'test']

data = pd.read_csv(os.path.join('/data/slue/', dataset, 'preprocessed','TroFi_formatted_all3737.csv'))

# random split
msk = np.random.rand(len(data)) < 0.9
train = data[msk]
devtest = data[~msk]
msk = np.random.rand(len(devtest)) < 0.5
dev = devtest[msk]
test = devtest[~msk]
print(len(train), len(dev), len(test))

# write to files
for dt_out,d in zip(dts_out, [train, dev, test]):
    fout = open(os.path.join('/data/slue',dataset,'{}.tsv'.format(dt_out)),'w')
    for index,row in d.iterrows():
        fout.write('{}\t{}\t{}\n'.format(index,'{} {}'.format(row.sentence,row.verb), row.label))
    fout.close()



# #for dt,dt_out in zip(dts,dts_out):
# for dt, dt_out in zip(dts, dts_out):

    # data = pd.read_csv(os.path.join('/data/slue',dataset,'preprocessed','VUA_formatted_{}.csv'.format(dt)),index_col=0, encoding = "ISO-8859-1")


    # fout = open(os.path.join('/data/slue',dataset,'{}.tsv'.format(dt_out)),'w')
    # for index, row in data.iterrows():
        # fout.write('{}\t{}\t{}\n'.format(index,'{} {}'.format(row.sentence,row.verb), row.label))
    # fout.close()
    # # for d,e in zip(dial,emo):
    # #     fout.write('{}\t{}\n'.format(d.strip(),e.strip()))
 #    # fout.close()

