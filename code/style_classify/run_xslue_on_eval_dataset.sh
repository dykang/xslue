#!/bin/bash


SLUE_DIR=/data/slue
SLUE_MODEL_DIR=/data/slue_model
TASK_NAMES=("PASTEL_country" "SARC" "SARC_pol"  "StanfordPoliteness" "GYAFC" "SarcasmGhosh" "DailyDialog" "ShortRomance" "CrowdFlower" "VUA" "TroFi" "ShortHumor" "ShortJokeKaggle" "HateOffensive" "PASTEL_politics" "PASTEL_tod" "PASTEL_age" "PASTEL_education" "PASTEL_ethnic" "PASTEL_gender" "SentiTreeBank" "EmoBank_v"  "EmoBank_a" "EmoBank_d" )

EVAL_DIR=/data/slue/diagnostic/
EVAL_DATASETS=("diagnostic_tweet_diversity_text.tsv") #"diagnostic_cross_test_text.tsv" "

for EVAL_DATASET in "${EVAL_DATASETS[@]}"
do
for TASK_NAME in "${TASK_NAMES[@]}"
do
    echo "Running ... ${TASK_NAME} ${EVAL_DATASET}"
    CUDA_VISIBLE_DEVICES=0 \
    python classify_bert.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name ${TASK_NAME} \
        --do_eval --eval_dataset ${EVAL_DIR}/${EVAL_DATASET} \
        --do_lower_case \
        --data_dir ${SLUE_DIR}/${TASK_NAME} \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=16   \
        --learning_rate 2e-5 \
        --output_dir ${SLUE_MODEL_DIR}/${TASK_NAME}/ \
        --overwrite_output_dir
done
done
