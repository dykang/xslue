import os
import glob
import re
import pandas as pd
import numpy as np
import random

def read_negative_samples():
    sents_neg = []

    with open('/data/slue/SARC/train.tsv') as f:
        for line in f:
            tks = line.strip().split('\t')
            if int(tks[1]) == 0:
                sents_neg.append((tks[2], 'SARC'))

    random.shuffle(sents_neg)
    print('Total number of sentences from literal@SARC',len(sents_neg))
    return sents_neg

data_neg = read_negative_samples()

dataset = 'ShortRomance'
dts_out = [ 'train', 'dev', 'test']

data = open(os.path.join('/data/slue/', dataset, 'crawled.txt')).readlines()
print(len(data))
data = [d.strip() for d in data]
new_data = []
for d in data:
    d = re.sub(r'\([^)]*\)', '', d)
    d = d.replace('  ',' ')
    new_data.append(d)

data = list(set(new_data))
print(len(data))

assert len(data) <= len(data_neg)
data = np.array(data)
data_neg = np.array(data_neg[:len(data)])

# random split
msk = np.random.rand(len(data)) < 0.9
train = data[msk]
train_neg = data_neg[msk]
devtest = data[~msk]
devtest_neg = data_neg[~msk]

msk = np.random.rand(len(devtest)) < 0.5
dev = devtest[msk]
dev_neg = devtest_neg[msk]
test = devtest[~msk]
test_neg = devtest_neg[~msk]
print(len(train), len(dev), len(test))
print(len(train_neg), len(dev_neg), len(test_neg))


# write to files
for dt_out,d,d_neg in zip(dts_out, [train, dev, test] , [train_neg, dev_neg,test_neg]):
    lines = []
    cnt = 0
    for index, row in enumerate(d):
        lines.append('{}\t{}\t{}'.format('web', 1, row))
    for line,source in d_neg:
        lines.append('{}\t{}\t{}'.format(source, 0, line))
    random.shuffle(lines)

    fout = open(os.path.join('/data/slue',dataset,'{}.tsv'.format(dt_out)),'w')
    for idx,line in enumerate(lines):
        fout.write('{}\t{}\n'.format(idx,line))
        cnt += 1
    fout.close()
    print(dt_out,cnt)



