function run(){
DATASET=race
RUN_NAME=alpha${ALPHA}
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B 
FLAG=ave
SUBSET=high
OUTPUT_DIR=$FLAG/$DATASET/$SUBSET/$MODEL
mkdir -p $OUTPUT_DIR
cp race_eval_high.sh $OUTPUT_DIR/run.sh

cd ..

python -u race_eval.py \
    --model-name /root/models/llama-7b \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --file_path ./head_config/race_top_high_70.json\
    --subset_name $SUBSET \
    --flag $FLAG \
    --alpha 1.3 \

cd experiments
}

run 

#llama-7b  1.3
#llama2-7b-hf  0.7