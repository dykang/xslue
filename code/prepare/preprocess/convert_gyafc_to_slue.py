import os
import glob

dts = ['train','tune', 'test']
dts_out = [ 'train', 'dev', 'test']

domains =  ['Family_Relationships', 'Entertainment_Music']

for dt,dt_out in zip(dts,dts_out):
    fout = open(os.path.join('slue_data','GYAFC','{}.tsv'.format(dt_out)),'w')
    for domain in domains:
        for label in ['formal','informal']:
            with open(os.path.join('GYAFC_Corpus/',domain, dt,label)) as f:
                for line in f:
                    line = line.strip()
                    fout.write('{}\t{}\n'.format(line, label))
    fout.close()

