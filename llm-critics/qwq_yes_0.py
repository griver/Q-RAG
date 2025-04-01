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

def build_promt(val_fact, candt_fact, question, answer):
    prompt = f'''<|im_start|>user
You are given four pieces of text:
    - A Question.
    - An Answer (the response given to the Question).
    - A Validated Fact (a fact confirmed as correct in a previous step).
    - A Candidate Fact (a fact we want to evaluate).

Your task is to determine whether the Candidate Fact is relevant and important for answering the Question, considering the provided Answer. Note that even if the Answer can be produced without explicitly using the Candidate Fact, the Candidate Fact may still be essential if it provides supporting context or contributes to the underlying chain-of-thought leading to the Answer. If the Candidate Fact plays a role in connecting reasoning steps or confirming key details, respond with "Yes". If it is extraneous, irrelevant, or simply noise, respond with "No". Provide only the answer "Yes" or "No" without any additional explanation.

Examples:

[Inputs: 
    - Question: "When was the baseball team winning the world series in 2015 baseball created?"
    - Answer: "1969"
    - Validated Fact: "The Kansas City Royals are an American professional baseball team based in Kansas City, Missouri. The Royals compete in Major League Baseball (MLB) as a member team of the American League (AL) Central division. The team was founded as an expansion franchise in 1969, and has participated in four World Series, winning in 1985 and 2015, and losing in 1980 and 2014."
    - Candidate Fact: "The 2015 World Series was the championship series of Major League Baseball's (MLB) 2015 season. The 111th edition of the World Series, it was a best - of - seven playoff between the National League (NL) champion New York Mets and the American League (AL) champion Kansas City Royals. The series was played between October 27 and November 1, with the Royals winning the series 4 games to 1. It was the first time since the 2010 World Series that the World Series extended into November. The Royals became the first team since the Oakland Athletics in the 1989 World Series to win the World Series after losing in the previous year. It was the first World Series to feature only expansion teams and the first since the 2007 World Series to not feature the Philadelphia Phillies, St. Louis Cardinals, or San Francisco Giants as the NL champions."
YOUR ANSWER: Yes
Explanation - | 'question': 'who won the world series in 2015 baseball', 'answer': 'Kansas City Royals', 'question': 'When was #1 created?', 'answer': '1969' | ], 

[Inputs:
    - Question: "Which major Russian city borders the body of water in which Saaremaa is located?"
    - Answer: "Saint Petersburg"
    - Validated Fact: "Since May 2004, with the accession of the Baltic states and Poland, the Baltic Sea has been almost entirely surrounded by countries of the European Union (EU). The only remaining non-EU shore areas are Russian: the Saint Petersburg area and the exclave of the Kaliningrad Oblast."
    - Candidate Fact: "The Oeselians or Osilians (Estonian saarlased; singular: saarlane) were a historical subdivision of Estonians inhabiting Saaremaa (Danish: Øsel; German: Ösel; Swedish: Ösel), an Estonian island in the Baltic Sea. They were first mentioned as early as the second century BC in Ptolemy's Geography III. The Oeselians were known in the Old Norse Icelandic Sagas and in Heimskringla as Víkingr frá Esthland (Estonian Vikings). Their sailing vessels were called pirate ships by Henry of Latvia in his Latin chronicles written at the beginning of the 13th century."
YOUR ANSWER: Yes
Explanation - | 'question': 'Where is Saaremaa located?', 'answer': 'the Baltic Sea', 'question': 'which major russian city borders #1', 'answer': 'Saint Petersburg' | ],

[Inputs:
    - Question: "Jan Šindel's was born in what country?"
    - Answer: "Czech Republic"
    - Validated Fact: "Jan Šindel was born in the Bohemian town Hradec Králové probably in the 1370s. As a young man he came to Prague to study at Charles University. In 1395 or 1399 he became the Master of Arts at Prague University. In 1406 he worked at the parish school of the St. Nicolas Church in the Lesser Town of Prague. Later he worked as a teacher of mathematics in Vienna, where he also studied medicine. Then he came back to Prague and became the professor of astronomy at Charles University, where he became Doctor of Medicine and rector of the university in 1410."
    - Candidate Fact: "Hradec Králové (; ) is a city of the Czech Republic, in the Hradec Králové Region of Bohemia. The city's economy is based on food-processing technology, photochemical, EMS and IT. Traditional industries include musical instrument manufacturing – the best known being Petrof pianos. The University of Hradec Králové is located in the city, the University of Defense has its only medical faculty in Hradec Králové and Charles University in Prague also has its Faculty of Medicine in Hradec Králové and Faculty of Pharmacy there."
YOUR ANSWER: Yes
Explanation - | 'question': 'What is Jan Šindel's birthplace?', 'answer': 'Hradec Králové', 'question': '#1 >> country', 'answer': 'Czech Republic' | ]

Now evaluate without explanation:

Inputs:
    - Question: "{question}"
    - Answer: "{answer}"
    - Validated Fact: "{val_fact}"
    - Candidate Fact: "{candt_fact}"

YOUR ANSWER:<|im_end|>\n<|im_start|>assistant\n<think>\n'''
    return prompt

path = '/trinity/home/a.anokhin/rmt_other_datasets/data/dataloaders/data_sources/musique'
with open(path + '/musique_ans_v1.0_train.jsonl', 'r') as json_file:
    json_list = list(json_file)
    raw_tasks_train = [(json.loads(json_str), "train") for json_str in json_list]

with open(path + '/musique_ans_v1.0_dev.jsonl', 'r') as json_file:
    json_list = list(json_file)
    raw_tasks_dev = [(json.loads(json_str), "dev") for json_str in json_list]

part_raw_tasks_train = raw_tasks_train[:2000]

def generate_yes(promt, model, tokenizer):
    inputs = tokenizer(promt, return_tensors="pt", truncation=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=35000,
        temperature=0.6,
        top_p=0.95
    )
    
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs["sequences"][:, input_length:]
    
    answer = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    
    # Если сгенерировано меньше двух токенов, вернуть None для дополнительных данных
    if new_tokens.shape[1] < 2:
        penultimate_token = None
        probability_yes = None
    else:
        # Определяем предпоследний токен по индексу
        penultimate_token_id = new_tokens[0, -2]
        penultimate_token = tokenizer.decode(penultimate_token_id, skip_special_tokens=True)
        
        penultimate_logits = outputs.scores[-2][0]
        
        # Применяем softmax для получения распределения вероятностей
        probs = torch.softmax(penultimate_logits, dim=0)
        
        # Получаем id токена для строки "Yes" (без специальных токенов)
        yes_token_ids = tokenizer.encode("Yes", add_special_tokens=False)
        if not yes_token_ids:
            probability_yes = 0.0
        else:
            # Если слово "Yes" токенизируется в один токен, берем его id
            yes_token_id = yes_token_ids[0]
            probability_yes = probs[yes_token_id].item()

    # Функция возвращает кортеж: сгенерированный ответ, предпоследний токен и вероятность "Yes" для предпоследнего шага
    return answer, penultimate_token, probability_yes

def LLM_critic_another_with_answer(data, model, tokenizer):

    win_first_second = 0
    win_first_noise = 0
    win_second_first = 0
    win_second_noise = 0

    all_true = 0
    all_noise = 0

    for i in tqdm.tqdm(range(len(data))):

        if data[i][0]['id'] in ['2hop__63593_126904', '2hop__28482_46077', '2hop__144408_215084']:
            print(data[i][0]['id'])
            continue

        question = data[i][0]['question']
        answer = data[i][0]['answer']

        id_first_fact = data[i][0]['question_decomposition'][0]['paragraph_support_idx']
        first_fact = data[i][0]['paragraphs'][id_first_fact]['paragraph_text']
        id_second_fact = data[i][0]['question_decomposition'][1]['paragraph_support_idx']
        second_fact = data[i][0]['paragraphs'][id_second_fact]['paragraph_text']

        promt_first_second = build_promt(first_fact, second_fact, question, answer)

        pred, last_token, prob = generate_yes(promt_first_second, model, tokenizer)
        pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

        if pred != last_token:
            print('pred != last_token')
            prob = 0.0

        if pred == 'Yes': #or pred =='yes' or pred == 'Yes ' or pred == ' Yes':
            pred = True
        elif pred == 'No': # or pred =='no' or pred == 'No ' or pred =='no ' or pred == ' No':
            pred = False
        else:
            print(repr(pred))
            pred = False

        if pred:
            win_first_second +=1
        else:
            print(data[i][0]['id'])
        all_true += 1

        data[i][0]['LLM_pred_first_second'] = pred
        data[i][0]['LLM_pred_first_second_probability'] = prob

        print(f'First - {pred}, last_token - {repr(last_token)}, prob - {prob}')

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

            pred, last_token, prob = generate_yes(promt_first_noise, model, tokenizer)
            pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

            if pred != last_token:
                print('pred != last_token')
                prob = 0.0

            if pred == 'Yes':
                pred = True
            elif pred == 'No':
                pred = False
            else:
                print(repr(pred))
                pred = False

            if not pred:
                win_first_noise += 1
            else:
                print(data[i][0]['id'])
            all_noise += 1

            data[i][0]['LLM_pred_first_noise'] = pred
            data[i][0]['LLM_pred_first_noise_probability'] = prob
        else:
            print("Нет доступных параграфов для выбора.")

        print(f'First_noise - {pred}, last_token - {repr(last_token)}, prob - {prob}')


        promt_second_first = build_promt(second_fact, first_fact, question, answer)

        pred, last_token, prob = generate_yes(promt_second_first, model, tokenizer)
        pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

        if pred != last_token:
            print('pred != last_token')
            prob = 0.0

        if pred == 'Yes': #or pred =='yes' or pred == 'Yes ' or pred == ' Yes':
            pred = True
        elif pred == 'No': # or pred =='no' or pred == 'No ' or pred =='no ' or pred == ' No':
            pred = False
        else:
            print(repr(pred))
            pred = False

        if pred:
            win_second_first +=1
        else:
            print(data[i][0]['id'])

        data[i][0]['LLM_pred_second_first'] = pred
        data[i][0]['LLM_pred_second_first_probability'] = prob

        print(f'Second - {pred}, last_token - {repr(last_token)}, prob - {prob}')

        if filtered_paragraphs:
            j = random.choice(filtered_paragraphs)

            paragraph_text = j['paragraph_text']
            promt_second_noise = build_promt(second_fact, paragraph_text, question, answer)

            pred, last_token, prob = generate_yes(promt_second_noise, model, tokenizer)
            pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

            if pred != last_token:
                print('pred != last_token')
                prob = 0.0

            if pred == 'Yes':
                pred = True
            elif pred == 'No':
                pred = False
            else:
                print(repr(pred))
                pred = False

            if not pred:
                win_second_noise += 1
            else:
                print(data[i][0]['id'])

            data[i][0]['LLM_pred_second_noise'] = pred
            data[i][0]['LLM_pred_second_noise_probability'] = prob
        else:
            print("Нет доступных параграфов для выбора.")


        print(f'Second_noise - {pred}, last_token - {repr(last_token)}, prob - {prob}')

        if i % 100 == 0:
            print(win_first_second, win_first_noise, win_second_first, win_second_noise, all_true, all_noise)
            file_path = f"/trinity/home/a.anokhin/stage_2/experiments_with_promt/qwq_yes_0.json"

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(part_raw_tasks_train, f)

        print(win_first_second, win_first_noise, win_second_first, win_second_noise, all_true, all_noise)

    return [win_first_second, win_first_noise, win_second_first, win_second_noise, all_true, all_noise]

result = LLM_critic_another_with_answer(part_raw_tasks_train, model, tokenizer)
print(result)

import os
file_path = f"/trinity/home/a.anokhin/stage_2/experiments_with_promt/qwq_yes_0.json"

os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as f:
    json.dump(part_raw_tasks_train, f)