function run(){
KEY_NUM=${1:-2}
DATASET=race
MODEL=LLaMA_hf_7B
OUTPUT_PATH=/root/HICD/experiments/output/$DATASET/$MODEL


mkdir -p $OUTPUT_PATH
cd ..

python -u race_keys.py \
    --model-name /root/models/Llama-7b \
    --output-path $OUTPUT_PATH/keys.json \
    --output-token-path $OUTPUT_PATH/tokens.json \
    --data-path /root/dataset/race \
    --num-gpus 1 \
    --key-num $KEY_NUM \


cd experiments
}

run 4
