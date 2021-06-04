XSLUE_DATA_DIR="./data"

TASK_NAMES=("SentiTreeBank" "EmoBank" "SARC" "SARC_pol" "StanfordPoliteness" "GYAFC"  "DailyDialog" "SarcasmGhosh" "ShortRomance" "CrowdFlower" "VUA" "TroFi" "ShortHumor" "ShortJokeKaggle" "HateOffensive" "PASTEL" )


for TASK_NAME in "${TASK_NAMES[@]}"
do
    echo "Downloading $TASK"
    wget http://dongtae.lti.cs.cmu.edu/data/xslue_v0.1/${TASK_NAME}.zip -P ${XSLUE_DATA_DIR}/

    echo "Unzipping $TASK"
    unzip ${XSLUE_DATA_DIR}/${TASK_NAME}.zip -d ${XSLUE_DATA_DIR}/
done



