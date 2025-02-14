import re
import os
import json
import transformers
import torch
from tqdm import tqdm, trange
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile

from modeling import HICD


def load_jsonl(file_path):
    list_data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        for idx in range(len(data)):
            list_data.append(data[idx]["document"])

    return list_data


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--max_gpu_memory", type=int, default=27)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--output-path", type=str, default="./keys")
parser.add_argument("--output-token-path", type=str, default="./tokens")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--key-num", type=int, default=3)

args = parser.parse_args()
model_name = args.model_name
num_gpus = args.num_gpus
device = args.device


fp = args.data_path
list_data = load_jsonl(fp)


llm = HICD(model_name, device, num_gpus, args.max_gpu_memory)
stop_word_list = ["#Document#:", "#Pondering#:"]
llm.set_stop_words(stop_word_list)

context = []
context.append("The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an \"extremely sad example of an abandoned and neglected exotic pet\". Inspector Selina Chan said: \"It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. \"We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. \"Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006.\" She added it was illegal to release non-native species into the wild.")
context.append("The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a \"whole nation resolved to answer hatred with love\". \"Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian,\" he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. \"Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose,\" he said on Tuesday.")
context.append("Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: \"I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. \"But let's not forget that there's two other families who are still waiting for their loved ones to be returned.\" The building was due for demolition when it partially collapsed in February.")

key_words = []
token_ranges_list =[]
with torch.no_grad():
    for q in context:
        print("context:")
        print(q)
        decoded_text, token_ranges=llm.key_words(q, key_num=args.key_num)
        key_words.append(decoded_text)
        token_ranges_list.append(token_ranges)
        print("key word:")
        print(key_words[-1])

cnt = 0
with torch.no_grad():
    for sample in tqdm(list_data):
        decoded_text, token_ranges=llm.key_words(sample, key_num=args.key_num)
        key_words.append(decoded_text)
        token_ranges_list.append(token_ranges)
        cnt += 1

output_file = args.output_path
output_file_token = args.output_token_path
with open(output_file, 'w') as f:
    json.dump(key_words, f)
with open(output_file_token, 'w') as f:
    json.dump(token_ranges_list, f)
