import os
import glob
import pandas as pd
import numpy as np

dataset = 'SARC'

dts = ['train','dev', 'test']
dts_out = [ 'train', 'dev', 'test']
data_slue = '/data/slue'

domains = ['-v2','-v2-pol']


# # random split
# msk = np.random.rand(len(data)) < 0.9
# train = data[msk]
# devtest = data[~msk]
# msk = np.random.rand(len(devtest)) < 0.5
# dev = devtest[msk]
# test = devtest[~msk]
# print(len(train), len(dev), len(test))


# write to files
for domain in domains:
    for dt,dt_out in zip(dts, dts_out):
        print('Reading...',domain,dt)
        data = pd.read_csv(os.path.join(data_slue, dataset, 'preprocessed','sarc{}'.format(domain),'{}.txt'.format(dt)),index_col=0,header=None,sep='\t', lineterminator='\n')

        fout = open(os.path.join(data_slue,dataset,'{}{}.tsv'.format(dt_out,domain)),'w')
        for index, row in data.iterrows():
            fout.write('{}\t{}\t{}\n'.format(index,row[1], row[2]))
        fout.close()


# for dt,dt_out in zip(dts,dts_out):
    # fout = open(os.path.join('../slue_data',dataset,'{}.tsv'.format(dt_out)),'w')
    # for domain in domains:
        # for label in ['formal','informal']:
            # with open(os.path.join('../slue_data/', dataset, domain, dt,label)) as f:
                # for line in f:
                    # line = line.strip()
                    # fout.write('{}\t{}\n'.format(line, label))
 #    fout.close()

