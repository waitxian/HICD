function run(){
KEY_NUM=${1:-2}
DATASET=summarization
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B 
OUTPUT_PATH=/root/HICD/experiments/output/$DATASET/$MODEL

mkdir -p $OUTPUT_PATH
cd ..

python -u halusum_keys.py \
    --model-name /root/models/Llama-7b \
    --data-path /root/dataset/${DATASET}_data.json \
    --output-path $OUTPUT_PATH/keys.json \
    --output-token-path $OUTPUT_PATH/tokens.json \
    --num-gpus 1 \
    --key-num $KEY_NUM \

cd experiments
}

run 4
