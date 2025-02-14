function run(){
KEY_NUM=${1:-2}
DATASET=hellaswag
MODEL=LLaMA_hf_7B
OUTPUT_PATH=/HICD/experiments/output/$DATASET/$MODEL


mkdir -p $OUTPUT_PATH
cd ..

python -u hellaswag_keys.py \
    --model-name /path/models/Llama-7b \
    --output-path $OUTPUT_PATH/keys.json \
    --output-token-path $OUTPUT_PATH/tokens.json \
    --data-path /path/dataset/hellaswag \
    --num-gpus 1 \
    --key-num $KEY_NUM \

cd experiments
}

run 8
