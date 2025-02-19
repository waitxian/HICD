## HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models

This repository implements the HICD.

**Environment setup**

```
conda create -n HICD python=3.10
conda activate HICD
cd HICD
pip install -r requirements.txt 
pip install git+https://github.com/davidbau/baukit
cd transformers
pip install -e .
cd experiments
```

**HICD**

To run the evaluation for a specific task, execute the `[task]_eval.sh` script , like:

```
bash hellaswag_eval.sh
```

Below are the parameters that can be set:

* -`--flag`    Options: `"None", "Cut", "Ave"`      - `"Ave"`: Use the average attention map method.    - `"Cut"`: Directly cut heads.    - `"None"`: No processing. 

- `--output-path`    Specify the path where the results will be saved.
- `--file_path`    Provide the path to the inducing heads' JSON configuration file.

- `--data-path`    Specify the dataset path. 
- `--model-name`    Choose the model path. You can select between:  - `"Llama-7b"`  - `"llama2-7b-hf"` 
- `--head-config`    In the `head_config` folder, different tasks' inducing heads' JSON configurations are provided.
- `--alpha`: Contrast factor

Here are the dataset links for evaluation:  [HellaSwag Dataset](https://huggingface.co/datasets/Rowan/hellaswag) ,[RACE Dataset](https://huggingface.co/datasets/ehovy/race) , [OpenBookQA Dataset](https://huggingface.co/datasets/allenai/openbookqa) ,[Summarization Data](https://github.com/RUCAIBox/HaluEval/blob/main/data/summarization_data.json) ,[TruthfulQA Dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa/viewer/multiple_choice) , [Factor Dataset](https://github.com/AI21Labs/factor/tree/main/data)

**Getting inducing heads config**

If you want to retrieve the inducing heads' JSON by yourself, you can implement it based on the lm-evaluation-harness project.

```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 11fa0bf4394998634e6c6e0c9fc2fc8211415042
```

Then, replace or add the contents in the `models` and `tasks` folders from the `lm_eval` folder of the HICD project into the corresponding `models` and `tasks` folders in `lm-evaluation-harness/lm_eval`. Afterward, replace the contents of `lm_eval/base.py`, `lm_eval/evaluator.py` and `lm_eval/utils.py` with the corresponding implementations. Replace `lm-evaluation-harness/main.py` with `HICD/lm_eval/main.py`  .Once these changes are made, you can run the code to obtain the `0shot_hellaswag_ac.pkl` file.

```
python main.py --model llama  --model_args pretrained=/path/models/Llama-7b  --tasks hellaswag --head_importance_calc --save_importance_path path/to/0shot_hellaswag_ac.pkl --num_fewshot 0
```

By setting `adv=True` in `tasks/hellaswag.py`, you can regenerate the erroneous samples and rerun the process to obtain the `0shot_hellaswag_ac.pkl` file.Similar implementations for constructing erroneous samples can be applied to other tasks as well.

```
python main.py --model llama  --model_args pretrained=/path/models/Llama-7b  --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/llama7b/0shot_hellaswag_ad.pkl --num_fewshot 0
```

You can then run `HICD/lm_eval/scripts/get_inducing_heads.py` to obtain the inducing heads configuration.We provide the .pkl  file in `HICD/head_importance/`.

```
python get_inducing_heads.py --saved_head_importance_path_ac /path/to/0shot_hellaswag_ac.pkl --saved_head_importance_path_ad /path/to/0shot_hellaswag_ad.pkl --save_file_path /path/head_config/hellaswag_top_30.json
```

**Other experiments implementation**

* PASTA-based

 `--file_path_inner`: Specify the heads for attention steering. We directly use inducing heads to compare with the HICD method.

`--token_ranges`: Determine the token positions where attention needs to be changed. You can run `[task]_keys.sh` to obtain the corresponding `token.json` file for the task.

`--scale`: Scaling factor that determines the extent of the attention change.

* SH2-based

 `--pondering`: Specify as "hard". 

 `--keys-path`: Configuration file for low-information words. You can run `[task]_keys.sh` to obtain the corresponding `key.json` file for the task.









