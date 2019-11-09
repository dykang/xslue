import os
import glob
import pandas as pd
import numpy as np
import gzip
import random


############################################
# negative sampling
############################################
def parse_biassum(tokens):
    sents, sent = [], []
    for t in tokens:
        if t.lower() == '<s>':
            sent = []
        elif t.lower() == '</s>':
            if len(sent) > 0:
                sents.append(sent)
        else:
            sent.append(t)
    sents = [(' '.join(sent),'BiasSum') for sent in sents]
    return sents

def read_negative_samples():
    sents_neg = []
    with gzip.open('/data/slue/ShortHumor/preprocessed/reddit.all.gz','rt', encoding='utf-8') as f:
        for line in f:
            tks = str(line).strip().split('\t')
            if len(tks) != 2:
                print(tks)
                from pdb import set_trace; set_trace()
            body = tks[0]
            ss = parse_biassum(body.split(' '))
            sents_neg += ss
    # print('Total number of sentences in Reddit@BiasSum',len(sents_neg))
    # print(sents_neg[-2])

    # sents = []
    with open('/data/slue/SARC/train.tsv') as f:
        for line in f:
            tks = line.strip().split('\t')
            if int(tks[1]) == 0:
                sents_neg.append((tks[2], 'SARC'))

    random.shuffle(sents_neg)
    print('Total number of sentences in Reddit@BiasSum & literal@SARC',len(sents_neg))
    return sents_neg

def main():
    data_neg = read_negative_samples()

    dataset = 'ShortHumor'

    dts = ['train','validation', 'test']
    dts_out = [ 'train', 'dev', 'test']

    data = []
    with open(os.path.join('/data/slue/', dataset, 'preprocessed','shorttext.csv')) as f:
        for line in f:
            line = line.strip()
            tks=line.split('\t')
            if len(tks) != 5:
                if len(tks)==1:
                    data[-1][1] += ' ' + tks[0]
                else:
                    print(tks)
                continue
            data.append([tks[2], tks[4].strip()])

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
    for dt_out,d,d_neg in zip(dts_out, [train, dev, test] , [train_neg, dev_neg, test_neg]):
        lines = []
        cnt = 0
        for index, row in enumerate(d):
            lines.append('{}\t{}\t{}'.format(row[0], 1, row[1]))
        for line,source in d_neg:
            lines.append('{}\t{}\t{}'.format(source, 0, line))
        random.shuffle(lines)

        fout = open(os.path.join('/data/slue',dataset,'{}.tsv'.format(dt_out)),'w')
        for idx,line in enumerate(lines):
            fout.write('{}\t{}\n'.format(idx,line))
            cnt += 1
        fout.close()
        print(dt_out,cnt)

if __name__ == '__main__':
    main()


