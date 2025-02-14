function run(){
DATASET=TruthfulQA
MODEL=LLaMA_hf_7B # LLaMA2_hf_7B 
FLAG=ave
OUTPUT_DIR=$FLAG/$DATASET/$MODEL

mkdir -p $OUTPUT_DIR
cp tfqa_mc_eval.sh $OUTPUT_DIR/run.sh

cd ..

python -u tfqa_mc_eval.py \
    --model-name  /root/models/Llama-7b\
    --data-path /root/dataset/TruthfulQA/TruthfulQA.csv \
    --output-path experiments/$OUTPUT_DIR/output.json \
    --file_path ./head_config/truthfulqa_top_70_10.json \
    --flag $FLAG \

cd experiments
}

run 

