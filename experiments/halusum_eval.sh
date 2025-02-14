function run(){
RUN_NAME=alpha${ALPHA}
DATASET=summarization
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B 
FLAG=ave
OUTPUT_DIR=$FLAG/HaluEval-$DATASET/$MODEL

mkdir -p $OUTPUT_DIR
cp halusum_eval.sh $OUTPUT_DIR/run.sh

cd ..

python -u halusum_eval.py \
    --model-name /root/models/Llama-7b \
    --data-path /root/dataset/${DATASET}_data.json \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --num-gpus 1 \
    --file_path ./head_config/halusum_top_30.json \
    --flag $FLAG\
    --alpha 1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3 \

cd experiments
}
run

