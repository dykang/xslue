# xSLUE
Data and code for ["xSLUE: A Benchmark and Analysis Platform for \\Cross-Style Language Understanding and Evaluation
"](https://arxiv.org) by Dongyeop Kang and Eduard Hovy. If you have any questions, please contact to Dongyeop Kang (dongyeok@cs.cmu.edu).

We provide an online platform ([xSLUE.com](http://xslue.com/)) for cross-style language understanding and evaluation.
The benchmark contains 15 different styles and 23 classification tasks. For each task, we also provide the fine-tuned BERT classifier for further analysis. Our analysis shows that some styles are highly dependent on each other (e.g., impoliteness and offense), and some domains (e.g., tweets, political debates) are stylistically more diverse than the others (e.g., academic manuscripts).


## Citation

    @inproceedings{kang19arxiv_xslue,
        title = {xSLUE: A Benchmark and Analysis Platform for \\Cross-Style Language Understanding and Evaluation},
        author = {Dongyeop Kang and Eduard Hovy},
        booktitle = {https://arxiv.org},
        url = {https://arxiv.org},
        year = {2019}
    }

### Note
- The diagnostic set is only available upon request, since thie work is under review. We will publicly release it upon acceptance.
 -

### Installation
Please download the pre-processed dataset and fine-tuned BERT classifier for each style in the [task](http://xslue.com/) tab.


Every corpora has the same format of dataset as follow:
```
Dataset format:
[source sentences] \t [target sentences]
or
<s> I was at home .. </s> <s> It was rainy day ..</s> ... \t <s> Sleeping at home rainy day </s> ..
```
An example python script for loading each dataset is provided here
```
python example/data_load.py --dataset AMI
```


### `run_glue.py`: Fine-tuning on GLUE tasks for sequence classification

The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running anyone of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

You should also install the additional packages required by the examples:

```shell
pip install -r ./examples/requirements.txt
```

```shell
export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file 'eval_results.txt' in the specified output_dir. In case of MNLI, since there are two separate dev sets, matched and mismatched, there will be a separate output folder called '/tmp/MNLI-MM/' in addition to '/tmp/MNLI/'.

#### Fine-tuning XLNet model on the STS-B regression task

This example code fine-tunes XLNet on the STS-B corpus using parallel training on a server with 4 V100 GPUs.
Parallel training is a simple way to use several GPUs (but is slower and less flexible than distributed training, see below).

```shell
export GLUE_DIR=/path/to/glue

python ./examples/run_glue.py \
    --model_type xlnet \
    --model_name_or_path xlnet-large-cased \
    --do_train  \
    --do_eval   \
    --task_name=sts-b     \
    --data_dir=${GLUE_DIR}/STS-B  \
    --output_dir=./proc_data/sts-b-110   \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --gradient_accumulation_steps=1 \
    --max_steps=1200  \
    --model_name=xlnet-large-cased   \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=120
```

On this machine we thus have a batch size of 32, please increase `gradient_accumulation_steps` to reach the same batch size if you have a smaller machine. These hyper-parameters should result in a Pearson correlation coefficient of `+0.917` on the development set.

#### Fine-tuning Bert model on the MRPC classification task

This example code fine-tunes the Bert Whole Word Masking model on the Microsoft Research Paraphrase Corpus (MRPC) corpus using distributed training on 8 V100 GPUs to reach a F1 > 92.

```bash
python -m torch.distributed.launch --nproc_per_node 8 ./examples/run_glue.py   \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name MRPC \
    --do_train   \
    --do_eval   \
    --do_lower_case   \
    --data_dir $GLUE_DIR/MRPC/   \
    --max_seq_length 128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --output_dir /tmp/mrpc_output/ \
    --overwrite_output_dir   \
    --overwrite_cache \
```

Training with these hyper-parameters gave us the following results:

```bash
  acc = 0.8823529411764706
  acc_and_f1 = 0.901702786377709
  eval_loss = 0.3418912578906332
  f1 = 0.9210526315789473
  global_step = 174
  loss = 0.07231863956341798
```




### xSLUE Benmark
(please check [task] tab for more details in [BiasSum.com](http://biassum.com))
 - Formality GYAFC Not public [original](https://github.com/raosudha89/GYAFC-corpus) [classifier](https://github.com/dykang/xslue)


### Leaderboard
 - Please contact to Dongyeop if you like to add your cross-style system to the leaderboard.

### Acknolwedgements
 - our style classification code is based on huggingface's [transformers](https://github.com/huggingface/transformers) on GLUE tasks.
 - our BiLSTM baseline code is based on [Pytorch-RNN-text-classification](https://github.com/keishinkickback/Pytorch-RNN-text-classification).


