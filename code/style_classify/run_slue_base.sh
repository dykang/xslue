#!/bin/bash


SLUE_DIR=/data/slue
SLUE_MODEL_DIR=/data/slue_model

MODEL_NAMES=("bilstm")

TASK_NAMES=("EmoBank_v"  "EmoBank_a" "EmoBank_d" "TroFi" "SentiTreeBank" "SARC" "SARC_pol" "StanfordPoliteness" "GYAFC" "SarcasmGhosh" "DailyDialog" "ShortRomance" "CrowdFlower" "VUA"  "ShortHumor" "ShortJokeKaggle" "HateOffensive" "PASTEL_politics" "PASTEL_country" "PASTEL_tod" "PASTEL_age" "PASTEL_education" "PASTEL_ethnic" "PASTEL_gender")


# Bi-LSTM baseline
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
for TASK_NAME in "${TASK_NAMES[@]}"
do
    echo "Running ... ${TASK_NAME} ${MODEL_NAME}"
    CUDA_VISIBLE_DEVICES=0 \
    python classify_baseline.py \
        --model_type ${MODEL_NAME} \
        --model_name_or_path bert-base-uncased \
        --task_name ${TASK_NAME} \
        --do_eval --do_train \
        --do_lower_case \
        --data_dir ${SLUE_DIR}/${TASK_NAME} \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=32   \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ${SLUE_MODEL_DIR}/${TASK_NAME}/${MODEL_NAME}/ \
        --overwrite_output_dir --overwrite_cache
     exit
done
done
