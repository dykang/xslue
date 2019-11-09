'''
Format:
'id'
'annotation_id'
'per_annotation_id'
'persona'
'input.keywords'
'input.images'
'input.sentences'
'output.sentences'
'''

import os
import glob
import json
from collections import defaultdict
import operator
import random

data_dir = '../PASTEL/data/'
level = 'sentences' # 'sentences' or 'stories'

STYLE_ORDER = ['gender', 'country', 'age', 'ethnic', 'education', 'politics', 'tod']


data = []
for dt in ['train','valid','test']:
    d = []
    files = glob.glob(os.path.join(data_dir, level, dt)+'/*.json')
    for file in files:
        with open(file) as fin:
            d.append(json.load(fin))
    print('Number of files/loaded-data: {}/{} in {}'.format(len(files),len(d), dt))
    data.append(d)

train,valid,test = tuple(data)


# show per-persona sentences
def get_dic(data, verbose=False):
    # persona dictionaries
    combined_dict = defaultdict(lambda: defaultdict(list))
    controlled_dict = defaultdict(list)

    for pid,obj in enumerate(data):
        sents = obj['output.sentences']
        sents = [sents]
        styles = obj['persona']

        # (1) combined setting dictionary
        for s,v in styles.items():
            if v.strip() == '':
                v = 'Empty'
            combined_dict[s][v] += sents

        # (2) controlled setting dictionary
        style_ordered_tuple = tuple([styles[s] for s in STYLE_ORDER])
        assert len(style_ordered_tuple) == len(STYLE_ORDER)
        controlled_dict[style_ordered_tuple] += sents

    if verbose:
        # combined setting
        print ('Combined dictionary',combined_dict.keys())
        print ('Combined Combinations',' '.join(
            ['%s:%d'%(k,len(v)) for k,v in combined_dict.items()]))
        for style, value_dic in combined_dict.items():
            print ('\t target STYLE: %s' % (style))
            print ('\t\t unique value types: %d'%(len(value_dic)))
            print ('\t\t value Dist: ',' '.join(['%s:%d'%(k,len(v)) for k,v in value_dic.items()]))
        print()

        # controlled setting
        print ('Controlled dictionary: total combination of styles=',len(controlled_dict))
        for style_order, value_dic in list(controlled_dict.items())[:5]:
            print('\t combination:',style_order, len(value_dic))
        print('\t ...')
    return combined_dict, controlled_dict

dataset = 'PASTEL'
for data, dt_out in zip([train, valid, test], ['train', 'dev', 'test']):
    combined_dict, _ = get_dic(data, verbose=False)
    for style, data in combined_dict.items():
        print(style,len(data),data.keys())
        data = dict(data)
        new_lines = []
        for freq_class in sorted(data, key=lambda k: len(data[k]), reverse=True):
            if freq_class == 'Empty': continue
            sents = data[freq_class]
            print('\t',freq_class,len(sents))
            # from pdb import set_trace; set_trace()
            for idx, sent in enumerate(sents):
                new_lines.append('{}\t{}'.format(freq_class,sent))
        random.shuffle(new_lines)

        fout = open(os.path.join('/data/slue',dataset,'{}_{}.tsv'.format(dt_out, style)),'w')
        for idx,line in enumerate(new_lines):
            fout.write('{}\t{}\n'.format(idx, line))
        fout.close()


