function run(){
DATASET=openbookqa
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B 
FLAG=ave
OUTPUT_DIR=$FLAG/$DATASET/$MODEL
mkdir -p $OUTPUT_DIR
cp openbookqa_eval.sh $OUTPUT_DIR/run.sh

cd ..

python -u openbookqa_eval.py \
    --model-name /path/models/Llama-7b \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --file_path ./head_config/openbookqa_top_70.json \
    --alpha  0.8 \
    --flag $FLAG \

cd experiments
}


run 

#llama-7b  0.8
#llama2-7b-hf  1.2