function run(){
DATASET=hellaswag
MODEL=LLaMA_hf_7B
FLAG=ave
OUTPUT_DIR=$FLAG/$DATASET/$MODEL

mkdir -p $OUTPUT_DIR
cp hellaswag_eval.sh $OUTPUT_DIR/run.sh

cd ..

python -u hellaswag_eval.py \
    --model-name /root/models/Llama-7b\
    --data-path /root/dataset/hellaswag \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --file_path /root/HICD/head_config/hellaswag_top_30.json \
    --flag $FLAG \

cd experiments
}

run 


