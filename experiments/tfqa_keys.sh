function run(){
KEY_NUM=${1:-2}
DATASET=TruthfulQA
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B
OUTPUT_PATH=/HICD/experiments/output/$DATASET/$MODEL

mkdir -p $OUTPUT_PATH
cd ..

python -u tfqa_keys.py \
    --model-name /path/models/Llama-7b \
    --output-path $OUTPUT_PATH/keys.json \
    --output-token-path $OUTPUT_PATH/tokens.json \
    --data-path  /path/dataset/TruthfulQA\
    --num-gpus 1 \
    --key-num $KEY_NUM \

cd experiments
}

run 10
