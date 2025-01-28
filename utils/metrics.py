from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np


class GenerativeAgent:

    def __init__(self, model, tokenizer, system_prompt, **default_gen_args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.default_gen_args = default_gen_args

        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, input_text, **generative_args):
        generative_args = dict(generative_args)
        for k,v in self.default_gen_args.items():
            if k not in generative_args:
                generative_args[k] = v
        messages = [{'role': 'system', 'content': self.system_prompt},
         {'role': 'user', 'content': input_text}]
        output = self.generate(messages, **generative_args)

        return output[0]['generated_text']


def compute_f1(model, tokenizer, input_texts, target_texts, system_prompt):
    # Подготовка входных данных
    inputs = [system_prompt + input_text for input_text in input_texts]

    # Получаем предсказания от модели
    model.eval()
    predictions = []

    for input_text in inputs:
        inputs_tokenized = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs_tokenized.input_ids, max_length=512, num_beams=5, no_repeat_ngram_size=2)

        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(predicted_text.strip())

    # Преобразуем target_texts и predictions в формат, подходящий для F1
    f1_scores = []
    for pred, target in zip(predictions, target_texts):
        pred_tokens = set(pred.split())
        target_tokens = set(target.split())

        # Рассчитываем F1 Score
        tp = len(pred_tokens & target_tokens)
        fp = len(pred_tokens - target_tokens)
        fn = len(target_tokens - pred_tokens)

        if tp + fp + fn == 0:
            f1_scores.append(1.0)  # если нет пересечений, считаем F1 равным 1
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

    # Среднее значение F1 Score
    return np.mean(f1_scores)


def compute_em(model, tokenizer, input_texts, target_texts, system_prompt):
    # Подготовка входных данных
    inputs = [system_prompt + input_text for input_text in input_texts]

    # Получаем предсказания от модели
    model.eval()
    correct_predictions = 0

    for input_text, target_text in zip(inputs, target_texts):
        inputs_tokenized = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs_tokenized.input_ids, max_length=512, num_beams=5, no_repeat_ngram_size=2)

        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Проверяем, совпадает ли предсказание с целью
        if predicted_text == target_text:
            correct_predictions += 1

    # Рассчитываем Exact Match
    em_score = correct_predictions / len(input_texts)
    return em_score