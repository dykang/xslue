import os
import glob
import pandas as pd
import numpy as np

dataset = 'EmoBank'

#dts = ['train','tune', 'test']
dts_out = [ 'train', 'dev', 'test']

data = pd.read_csv(os.path.join('../slue_data/', dataset, 'original','emobank.csv'),index_col=0)

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
    fout = open(os.path.join('../slue_data',dataset,'{}.tsv'.format(dt_out)),'w')
    for index, row in d.iterrows():
        fout.write('{}\t{}\t{}\t{}\t{}\n'.format(index,row.V, row.A, row.D, row.text))
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

