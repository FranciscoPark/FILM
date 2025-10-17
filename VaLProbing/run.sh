#!/bin/bash
# List of models
MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

export NCCL_IGNORE_DISABLED_P2P=1

# List of test files
TESTFILES=(
    "document_bi_32k.jsonl"
    "code_backward_32k.jsonl"
    "database_forward_32k.jsonl"
)

TESTDATA_FOLDER="/root/sky_workdir/FILM/VaLProbing/ValProbing-32K/"
OUTPUT_BASE="./VaLProbing-32K/results/"
MAXLEN=128
TP_SIZE=1

for MODEL in "${MODELS[@]}"; do
    echo "============================="
    echo "Running inference for $MODEL"
    echo "============================="

    for FILE in "${TESTFILES[@]}"; do
        echo "Processing $FILE ..."
        python ../vllm_inference/vllm_inference.py \
            --model_path "$MODEL" \
            --testdata_file "$FILE" \
            --testdata_folder "$TESTDATA_FOLDER" \
            --output_folder "${OUTPUT_BASE}${MODEL}/" \
            --max_length "$MAXLEN" \
            --tensor_parallel_size "$TP_SIZE"
    done
done