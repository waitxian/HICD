function run(){
RUN_NAME=alpha${ALPHA}
DATASET=summarization
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B  1
FLAG=ave
OUTPUT_DIR=$FLAG/HaluEval-$DATASET/$MODEL

mkdir -p $OUTPUT_DIR
cp halusum_eval.sh $OUTPUT_DIR/run.sh

cd ..

python -u halusum_eval.py \
    --model-name /path/models/Llama-7b \
    --data-path /path/dataset/${DATASET}_data.json \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
    --file_path ./head_config/halusum_top_30.json \
    --flag $FLAG\
    --alpha 2 \

cd experiments
}
run

