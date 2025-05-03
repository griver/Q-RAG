import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import torch.nn.functional as F
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import random
import os

model_name = "Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map='cuda',
    trust_remote_code=True
).to("cuda")

def generate(promt, model, tokenizer):
    inputs = tokenizer(promt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            max_new_tokens=35000,
            temperature=0.6,
            top_p=0.95
        )
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs["sequences"][:, input_length:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)

def build_promt(val_fact, candt_fact, question, answer):
    prompt = f'''<|im_start|>user
You are given four pieces of text:
    - A Question.
    - An Answer (the response given to the Question).
    - A Validated Fact (a fact confirmed as correct in a previous step).
    - A Candidate Fact (a fact we want to evaluate).

Your task is to determine the numerical rating for the Candidate Fact's relevance and importance in answering the Question, considering the provided Answer. Note that even if the Answer can be produced without explicitly using the Candidate Fact, the Candidate Fact may still be essential if it provides supporting context or contributes to the underlying chain-of-thought leading to the Answer. Rate the Candidate Fact on a scale from 0 to 10, where 0 indicates that the Candidate Fact is completely irrelevant or extraneous, and 10 indicates that it is absolutely crucial for producing the Answer. Provide only the numerical rating without any additional explanation.

Examples:

[Inputs:
    - Question: "Which major Russian city borders the body of water in which Saaremaa is located?"
    - Answer: "Saint Petersburg"
    - Validated Fact: "Since May 2004, with the accession of the Baltic states and Poland, the Baltic Sea has been almost entirely surrounded by countries of the European Union (EU). The only remaining non-EU shore areas are Russian: the Saint Petersburg area and the exclave of the Kaliningrad Oblast."
    - Candidate Fact: "The Oeselians or Osilians (Estonian saarlased; singular: saarlane) were a historical subdivision of Estonians inhabiting Saaremaa (Danish: Øsel; German: Ösel; Swedish: Ösel), an Estonian island in the Baltic Sea. They were first mentioned as early as the second century BC in Ptolemy's Geography III. The Oeselians were known in the Old Norse Icelandic Sagas and in Heimskringla as Víkingr frá Esthland (Estonian Vikings). Their sailing vessels were called pirate ships by Henry of Latvia in his Latin chronicles written at the beginning of the 13th century."
YOUR ANSWER: 10
Explanation - | 'question': 'Where is Saaremaa located?', 'answer': 'the Baltic Sea', 'question': 'which major russian city borders #1', 'answer': 'Saint Petersburg' | ],

[Inputs:
    - Question: "When was the baseball team winning the world series in 2015 baseball created?"
    - Answer: "1969"
    - Validated Fact: "The Kansas City Royals are an American professional baseball team based in Kansas City, Missouri. The Royals compete in Major League Baseball (MLB) as a member team of the American League (AL) Central division. The team was founded as an expansion franchise in 1969, and has participated in four World Series, winning in 1985 and 2015, and losing in 1980 and 2014."
    - Candidate Fact: "The 2015 World Series was the championship series of Major League Baseball's (MLB) 2015 season. The 111th edition of the World Series, it was a best - of - seven playoff between the National League (NL) champion New York Mets and the American League (AL) champion Kansas City Royals. The series was played between October 27 and November 1, with the Royals winning the series 4 games to 1. It was the first time since the 2010 World Series that the World Series extended into November. The Royals became the first team since the Oakland Athletics in the 1989 World Series to win the World Series after losing in the previous year. It was the first World Series to feature only expansion teams and the first since the 2007 World Series to not feature the Philadelphia Phillies, St. Louis Cardinals, or San Francisco Giants as the NL champions."
YOUR ANSWER: 10
Explanation - | 'question': 'who won the world series in 2015 baseball', 'answer': 'Kansas City Royals', 'question': 'When was #1 created?', 'answer': '1969' | ],

[Inputs:
    - Question: "When was the territory covered by RIBA's Cambridge branch office created?"
    - Answer: "1994"
    - Validated Fact: "The East of England is one of nine official regions of England at the first level of NUTS for statistical purposes. It was created in 1994 and was adopted for statistics from 1999. It includes the ceremonial counties of Bedfordshire, Cambridgeshire, Essex, Hertfordshire, Norfolk and Suffolk. Essex has the highest population in the region."
    - Candidate Fact: "The Institute also maintains a dozen regional offices around the United Kingdom, it opened its first regional office for the East of England at Cambridge in 1966."
YOUR ANSWER: 10
Explanation - | 'question': 'What territory did RIBA's Cambridge branch office cover?', 'answer': 'the East of England', 'question': 'When was #1 birthed?', 'answer': '1994' | ]

Now evaluate without explanation:

Inputs:
    - Question: "{question}"
    - Answer: "{answer}"
    - Validated Fact: "{val_fact}"
    - Candidate Fact: "{candt_fact}"

YOUR ANSWER:<|im_end|>\n<|im_start|>assistant\n<think>\n'''
    return prompt

def safe_int(value):
    try:
        return int(value)
    except ValueError:
        print(repr(value))
        return 0

def LLM_critic_score(data, model, tokenizer):

    for i in tqdm.tqdm(range(len(data))):

        if data[i][0]['id'] in ['2hop__28482_46077', '2hop__63593_126904', '2hop__7483_160863']:
            print(data[i][0]['id'])
            continue

        question = data[i][0]['question']
        answer = data[i][0]['answer']

        id_first_fact = data[i][0]['question_decomposition'][0]['paragraph_support_idx']
        first_fact = data[i][0]['paragraphs'][id_first_fact]['paragraph_text']
        id_second_fact = data[i][0]['question_decomposition'][1]['paragraph_support_idx']
        second_fact = data[i][0]['paragraphs'][id_second_fact]['paragraph_text']

        promt_first_second = build_promt(first_fact, second_fact, question, answer)

        pred = generate(promt_first_second, model, tokenizer)
        pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

        data[i][0]['LLM_pred_first_second_score'] = safe_int(pred)

        print(f'First - {pred}')

        # Фильтруем параграфы, исключая те, у которых idx равен id_first_fact или id_second_fact
        filtered_paragraphs = [
            j for j in data[i][0]['paragraphs']
            if j['idx'] != id_first_fact and j['idx'] != id_second_fact
        ]

        # Если после фильтрации остались параграфы, выбираем один случайный
        if filtered_paragraphs:
            j = random.choice(filtered_paragraphs)

            paragraph_text = j['paragraph_text']
            promt_first_noise = build_promt(first_fact, paragraph_text, question, answer)

            pred = generate(promt_first_noise, model, tokenizer)
            pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

            data[i][0]['LLM_pred_first_noise_score'] = pred
        else:
            print("Нет доступных параграфов для выбора.")

        print(f'First_noise - {pred}')


        promt_second_first = build_promt(second_fact, first_fact, question, answer)

        pred = generate(promt_second_first, model, tokenizer)
        pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

        data[i][0]['LLM_pred_second_first_score'] = pred

        print(f'Second - {pred}')

        if filtered_paragraphs:
            j = random.choice(filtered_paragraphs)

            paragraph_text = j['paragraph_text']
            promt_second_noise = build_promt(second_fact, paragraph_text, question, answer)

            pred = generate(promt_second_noise, model, tokenizer)
            pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

            data[i][0]['LLM_pred_second_noise_score'] = pred
        else:
            print("Нет доступных параграфов для выбора.")


        print(f'Second_noise - {pred}')

        if i % 100 == 0:
            file_path = f"/trinity/home/a.anokhin/stage_2/experiments_with_promt/qwq_score_0.json"

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(part_raw_tasks_train, f)


path = '/trinity/home/a.anokhin/rmt_other_datasets/data/dataloaders/data_sources/musique'
with open(path + '/musique_ans_v1.0_train.jsonl', 'r') as json_file:
    json_list = list(json_file)
    raw_tasks_train = [(json.loads(json_str), "train") for json_str in json_list]


part_raw_tasks_train = raw_tasks_train[:2000]

LLM_critic_score(part_raw_tasks_train, model, tokenizer)

file_path = f"/trinity/home/a.anokhin/stage_2/experiments_with_promt/qwq_score_0.json"

os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as f:
    json.dump(part_raw_tasks_train, f)