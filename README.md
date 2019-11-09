# xSLUE
Data and code for ["xSLUE: A Benchmark and Analysis Platform for \\Cross-Style Language Understanding and Evaluation
"](https://arxiv.org) by Dongyeop Kang and Eduard Hovy. If you have any questions, please contact to Dongyeop Kang (dongyeok@cs.cmu.edu).

We provide an online platform ([http://xslue.com/](http://xslue.com/)) for cross-style language understanding and evaluation.
The [Cross-Style Language Understanding and Evaluation (xSLUE) benchmark](https://xslue.com/) contains 15 different styles and 23 classification tasks. For each task, we also provide the fine-tuned BERT classifier for further analysis. Our analysis shows that some styles are highly dependent on each other (e.g., impoliteness and offense), and some domains (e.g., tweets, political debates) are stylistically more diverse than the others (e.g., academic manuscripts).


## Citation
    @inproceedings{kang19arxiv_xslue,
        title = {xSLUE: A Benchmark and Analysis Platform for Cross-Style Language Understanding and Evaluation},
        author = {Dongyeop Kang and Eduard Hovy},
        booktitle = {https://arxiv.org},
        url = {https://arxiv.org},
        year = {2019}
    }

### Note
- The diagnostic set is only available upon request, since this work is under review. We will publicly release it upon acceptance. However, we will evaluate your system on another, private diagnostic set on cross-style classification and report the score in the leaderboard.


### `run_xslue.sh`: Fine-tuning on xSLUE tasks for style classification

Before running any xSLUE tasks you should download the
[xSLUE data](https://xslue.com) by running this script (TODO) and unpack it to some directory `$XSLUE_DIR`.

Please download the pre-processed dataset and fine-tuned BERT classifier for each style in the [task](http://xslue.com/) tab. Or, you can download them in the table below.


An example python script for loading each dataset is provided here
```shell
cd code/style_classify/
./run_xslue.sh
```

You should also install the additional packages required by the examples:

```shell
pip install -r ./requirements.txt
```

or

```shell
SLUE_DIR=$HOME/data/slue
SLUE_MODEL_DIR=$HOME/data/slue_model

TASK_NAMES=("SentiTreeBank" "EmoBank_v"  "EmoBank_a" "EmoBank_d" "SARC" "SARC_pol" "StanfordPoliteness" "GYAFC"  "DailyDialog" "SarcasmGhosh" "ShortRomance" "CrowdFlower" "VUA" "TroFi" "ShortHumor" "ShortJokeKaggle" "HateOffensive" "PASTEL_politics" "PASTEL_country" "PASTEL_tod" "PASTEL_age" "PASTEL_education" "PASTEL_ethnic" "PASTEL_gender")

MODEL=bert-base-uncased

for TASK_NAME in "${TASK_NAMES[@]}"
do
    echo "Running ... ${TASK_NAME}"
    CUDA_VISIBLE_DEVICES=0 \
    python classify_bert.py \
        --model_type bert \
        --model_name_or_path ${MODEL} \
        --task_name ${TASK_NAME} \
        --do_eval --do_train \
        --do_lower_case \
        --data_dir ${SLUE_DIR}/${TASK_NAME} \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ${SLUE_MODEL_DIR}/${TASK_NAME}/${MODEL}/ \
        --overwrite_output_dir --overwrite_cache
done
```


### xSLUE Benmark
(please check [task] tab for more details in [BiasSum.com](http://biassum.com))
 - Formality GYAFC Not public [original](https://github.com/raosudha89/GYAFC-corpus) [classifier](https://github.com/dykang/xslue)


### Leaderboard
 - Please contact to Dongyeop (dongyeok@cs.cmu.edu) if you like to add your cross-style system to the leaderboard. We will be testing your model on our private diagnostic set as well. 

### Acknolwedgements
 - our style classification code is based on huggingface's [transformers](https://github.com/huggingface/transformers) on GLUE tasks.
 - the structure of our benchmark and basic idea are motiviated by [GLUE](https://gluebenchmark.com/) project.
 - our BiLSTM baseline code is based on [Pytorch-RNN-text-classification](https://github.com/keishinkickback/Pytorch-RNN-text-classification).


