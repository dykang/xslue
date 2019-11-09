#!/bin/bash


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

        #  --eval_dataset ${EVAL_DATASET}
    # exit
done
