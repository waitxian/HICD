function run(){
DATASET=news_factor
MODEL=LLaMA_hf_7B
FLAG=ave
OUTPUT_DIR=$FLAG/$DATASET/$MODEL

mkdir -p $OUTPUT_DIR
cp factor_eval.sh $OUTPUT_DIR/run.sh

cd ..

python -u factor_eval.py \
    --model-name /path/models/Llama-7b \
    --data-path /path/dataset/$DATASET.csv \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --file_path ./head_config/factor_top_70_10.json \
    --alpha 0.38 \
    --flag $FLAG \

cd experiments
}

run 

#llama-7b  0.38
#llama2-7b-hf  0.38
