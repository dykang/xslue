from __future__ import absolute_import, division, print_function

import argparse
import glob
from collections import defaultdict
import os
import random
import coloredlogs, logging
import numpy as np
from tqdm import tqdm, trange
import torch

from utils_classify import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',level='INFO',datefmt='%m/%d %H:%M:%S',logger=logger)

from classify import ALL_MODELS,MODEL_CLASSES

def calculate_data_stats(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels()
    num_labels = len(label_list)


    # Load data features from cache or dataset file

    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

    train = processor.get_train_examples(args.data_dir)
    dev = processor.get_dev_examples(args.data_dir)
    test = processor.get_test_examples(args.data_dir)

    examples = train + dev + test
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)


    print(task, processor)
    # sentence length
    print('Total files:',len(features))
    nonzero_cnt = 0
    for feature in features:
        nonzero_cnt += (np.count_nonzero(feature.input_ids) - 2)
    print('Average sentence length: {:.2f}'.format(nonzero_cnt / len(features)))

    # Vocab size
    vocab_dict = defaultdict(int)
    for feature in features:
        for f in feature.input_ids:
            vocab_dict[f] += 1
    print('Vocab size:',len(vocab_dict)-2)

    # feature distribution
    feature_dict = defaultdict(int)
    for feature in features:
        feature_dict[feature.label_id] += 1
    print('Feature distribution:')
    total_c = sum([c for _,c in feature_dict.items()])
    for f,c in feature_dict.items():
        print('\t{}\t{}\t{:.2f}'.format(label_list[f], c, 100.0 * c /total_c ))
    print()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,                         help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")



    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")


    # # added for external input for testing
    # parser.add_argument("--eval_dataset", default=None, type=str,
    #                     help="Additional eval dataset ")


    args = parser.parse_args()

    # subtask partition: e.g., SARC_pol -> SARC in data_dir
    args.data_dir = args.data_dir.split('_')[0]

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info("Processor: {}, label: {} ({})".format(processor,label_list,num_labels))


    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)



    # Training
    calculate_data_stats(args, args.task_name, tokenizer, evaluate=False)



if __name__ == "__main__":
    main()
