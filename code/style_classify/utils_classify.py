""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_text(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines





def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if example.label is None:
            label_id = None
        else:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s ({} = {})".format(example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }



# added for SLUE
class GyafcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    """Processor for the GYAFC data set (SLUE version)."""
    def get_labels(self):
        """See base class."""
        return ["formal", "informal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                text_a = line[0]
                if len(line) == 2:
                    label = line[1]
                else:
                    label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StanfordPolitenessProcessor(DataProcessor):
    """Processor for the StanfordPoliteness data set (SLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["polite", "impolite"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # if len(line) != 4:
            #     continue
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[2]
                    if len(line) == 4:
                        label = 'polite' if float(line[3]) > 0 else 'impolite'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples





class SARC(DataProcessor):
    """Processor for the SARCv2 data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["sarcastic", "literal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)

            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[2]
                    if len(line) >= 2:
                        label = 'sarcastic' if line[1] == '1' else 'literal'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SARCpol(SARC):
    """Processor for the SARCv2 data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train-pol.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev-pol.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test-pol.tsv")), "test")



class DailyDialog(DataProcessor):
    """Processor for the SARCv2 data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["noemotion","anger","disgust","fear","happiness","sadness","surprise"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)

            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[0]
                    if len(line) >= 2:
                        label = self.get_labels()[int(line[1])]
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class CrowdFlower(DataProcessor):
    """Processor for the CrowdFlower data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["empty", "sadness", "enthusiasm", "neutral", "worry", "sadness", "love", "fun", "hate", "happiness", "relief", "boredom", "surprise", "anger"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[2]
                    if len(line) >= 2:
                        label = line[1]
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples




class EmoBank(DataProcessor):
    """Processor for the SARCv2 data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return [None] #"valence", "arousal", "dominance"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        return

class EmoBank_v(EmoBank):
    def get_labels(self):
        """See base class."""
        return ['valence(positive)'] #"valence", "arousal", "dominance"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                text_a = line[4]
                if len(line) >= 4:
                    label = line[1] # [line[1], line[2], line[3]]
                else:
                    label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class EmoBank_a(EmoBank):
    def get_labels(self):
        """See base class."""
        return ['arousal(excited)'] #"valence", "arousal", "dominance"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                text_a = line[4]
                if len(line) >= 4:
                    label = line[2] # [line[1], line[2], line[3]]
                else:
                    label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class EmoBank_d(EmoBank):
    def get_labels(self):
        """See base class."""
        return ['dominance(being_in_control)'] #"valence", "arousal", "dominance"]


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:

                text_a = line[4]
                if len(line) >= 4:
                    label = line[3] # [line[1], line[2], line[3]]
                else:
                    label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



class SarcasmGhosh(DataProcessor):
    """Processor for the SarcasmGhosh data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["sarcastic","literal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:

                try:
                    text_a = line[2]
                    if len(line) >= 2:
                        label = 'sarcastic' if line[1] == '1' else 'literal'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ShortHumor(DataProcessor):
    """Processor for the ShortHumor data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["humorous","literal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:

                try:
                    text_a = line[3]
                    if len(line) >= 3:
                        label = 'humorous' if line[2] == '1' else 'literal'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ShortJokeKaggle(DataProcessor):
    """Processor for the ShortJokeKaggle data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["humorous","literal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[3]
                    if len(line) >= 3:
                        label = 'humorous' if line[2] == '1' else 'literal'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples





class PASTEL(DataProcessor):
    """Processor for the PASTEL data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[2]
                    if len(line) >= 2:
                        label = line[1] # self.get_labels()[line[1]]
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class PASTEL_politics(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_politics.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_politics.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_politics.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["LeftWing", "Centrist", "RightWing"]

class PASTEL_country(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_country.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_country.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_country.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["U.S.A", "U.K"]

class PASTEL_politics(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_politics.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_politics.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_politics.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["LeftWing", "Centrist", "RightWing"]

class PASTEL_tod(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_tod.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_tod.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_tod.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["Midnight", "Night", "Afternoon", "Morning", "Evening"]

class PASTEL_age(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_age.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_age.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_age.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["25-34", "55-74", "35-44", "18-24", "Under12", "45-54", "75YearsOrOlder", "12-17"]

class PASTEL_education(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_education.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_education.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_education.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["Bachelor", "Master", "TradeOrTechnicalOrVocationalTraining", "HighSchool","NoDegree", "Edu-AssociateDegree", "Doctorate", "NoSchoolingCompleted", "SomeHighSchool,NoDiploma","NurserySchoolTo8thGrade"]

class PASTEL_ethnic(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_ethnic.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_ethnic.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_ethnic.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["MiddleEastern", "Ethnic-NativeAmerican", "Caucasian", "HispanicOrLatino", "African", "Other", "EastAsian", "SouthAsian", "PacificIslander", "Ethnic-CentralAsian"]

class PASTEL_gender(PASTEL):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_gender.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_gender.tsv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_gender.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["Female", "Male", "Non-binary"]



class HateOffensive(DataProcessor):
    """Processor for the HateOffensive data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["hate", "offensive", "neither"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[2]
                    if len(line) >= 2:
                        label = 'hate' if line[1]=='0' else 'offensive' if line[1]=='1' else 'neither'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



class TroFi(DataProcessor):
    """Processor for the TroFi data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["metaphor", "non-metaphor"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:

                try:
                    text_a = line[1]
                    if len(line) >= 2:
                        label = 'metaphor' if line[2]=='1' else 'non-metaphor'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class VUA(DataProcessor):
    """Processor for the VUA data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["metaphor", "non-metaphor"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:

                try:
                    text_a = line[1]
                    if len(line) >= 2:
                        label = 'metaphor' if line[2]=='1' else 'non-metaphor'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



class ShortRomance(DataProcessor):
    """Processor for the ShortRomance data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["romantic", "literal"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:

                try:
                    text_a = line[3]
                    if len(line) >= 2:
                        label = 'romantic' if line[2]=='1' else 'literal'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



class SentiTreeBankProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    """Processor for the Sentiment TreeBank data set (SLUE version)."""
    """We make coarse-grained version (e.g., positive, negative), which can be extended to fine-grained one easily"""
    def get_labels(self):
        """See base class."""
        # return ["very positive", "positive", "neutral", "negative", "very negative"]
        return ["positive", "negative"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = self.get_labels()
        for (i, line) in enumerate(lines):
            if i == 0 and len(line) != 1:
                continue
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                if line[4] not in labels:
                    continue
                text_a = line[1]
                if len(line) == 5:
                    label = line[4]
                else:
                    label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



processors = {
    "gyafc": GyafcProcessor,
    "stanfordpoliteness": StanfordPolitenessProcessor,
    "sarc": SARC,
    "sarc_pol": SARCpol,

    "dailydialog": DailyDialog,
    "emobank_v": EmoBank_v,
    "emobank_a": EmoBank_a,
    "emobank_d": EmoBank_d,
    "crowdflower": CrowdFlower,

    "sarcasmghosh": SarcasmGhosh,
    "shorthumor": ShortHumor,
    "shortjokekaggle": ShortJokeKaggle,

    "pastel_country": PASTEL_country,
    "pastel_politics": PASTEL_politics,
    "pastel_tod": PASTEL_tod,
    "pastel_age": PASTEL_age,
    "pastel_education": PASTEL_education,
    "pastel_ethnic": PASTEL_ethnic,
    "pastel_gender": PASTEL_gender,

    "hateoffensive": HateOffensive,
    "trofi": TroFi,
    "vua": VUA,

    "shortromance": ShortRomance,
    "sentitreebank": SentiTreeBankProcessor,
}

output_modes = {
    "gyafc": "classification",
    "stanfordpoliteness": "classification",
    "sarc": "classification",
    "sarc_pol": "classification",

    "dailydialog": "classification",
    "emobank_v": "regression",
    "emobank_a": "regression",
    "emobank_d": "regression",
    "crowdflower": "classification",

    "sarcasmghosh": "classification",
    "shorthumor": "classification",
    "shortjokekaggle": "classification",

    "pastel_country": "classification",
    "pastel_politics": "classification",
    "pastel_tod": "classification",
    "pastel_age": "classification",
    "pastel_education": "classification",
    "pastel_ethnic": "classification",
    "pastel_gender": "classification",

    "hateoffensive": "classification",
    "trofi": "classification",
    "vua": "classification",

    "shortromance": "classification",
    "sentitreebank": "classification",
}

SLUE_TASKS_NUM_LABELS = {
    "gyafc": 2,
    "stanfordpoliteness": 2,
    "sarc": 2,
    "sarc_pol": 2,

    "dailydialog": 7,
    "emobank_v": 1,
    "emobank_a": 1,
    "emobank_d": 1,
    "crowdflower": 14,

    "sarcasmghosh": 2,
    "shorthumor": 2,
    "shortjokekaggle": 2,

    "pastel_country": 2,
    "pastel_politics": 3,
    "pastel_tod": 5,
    "pastel_age": 8,
    "pastel_education": 10,
    "pastel_ethnic": 10,
    "pastel_gender": 3,

    "hateoffensive": 2,
    "trofi": 2,
    "vua": 2,

    "shortromance":2,
    "sentitreebank": 2,
}



def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    # added for SLUE
    if task_name == "gyafc":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "stanfordpoliteness":
        return acc_and_f1(preds, labels)
        #return {"acc": simple_accuracy(preds, labels)}
    elif task_name in [ "sarc", "sarc_pol"]:
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "dailydialog":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name.startswith("emobank"):
        return pearson_and_spearman(preds, labels)
    elif task_name == "crowdflower":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sarcasmghosh":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "shorthumor":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "shortjokekaggle":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name.startswith("pastel"):
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hateoffensive":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "trofi":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "vua":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "shortromance":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sentitreebank":
        return acc_and_f1(preds, labels)
        # return {"acc": simple_accuracy(preds, labels)}

    else:
        raise KeyError(task_name)









