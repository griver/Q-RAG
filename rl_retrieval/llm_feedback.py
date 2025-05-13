from __future__ import annotations
from abc import ABC, abstractmethod
import re, string
from typing import Callable, List, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.manual_seed(42)

from .feedback import AFeedbackModel


class GenerationMixin:
    def __init__(self, gen_model_name: str, device_gen: str = "auto"):
        # Load generator model and tokenizer
        self.gen_tok = AutoTokenizer.from_pretrained(gen_model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_gen,
            trust_remote_code=True
        )

    def _build_prompt_gen(self, passages, question):
        context = "\n".join(passages)
        prompt = f"""You need to answer a question. There is a context which is crucial to answer question. There may be noisy facts in the context that are not relevant, keep this in mind and be careful!

Examples:

[Inputs: 
    - Question: "In what year was the author of The Insider's Guide to the Colleges established?"
    - Context: "TITLE: Yale University\nTEXT: The university hosts a variety of student journals, magazines, and newspapers. Established in 1872, The Yale Record is the world\'s oldest humor magazine. Newspapers include the Yale Daily News, which was first published in 1878, and the weekly Yale Herald, which was first published in 1986. Dwight Hall, an independent, non-profit community service organization, oversees more than 2,000 Yale undergraduates working on more than 70 community service initiatives in New Haven. The Yale College Council runs several agencies that oversee campus wide activities and student services. The Yale Dramatic Association and Bulldog Productions cater to the theater and film communities, respectively. In addition, the Yale Drama Coalition serves to coordinate between and provide resources for the various Sudler Fund sponsored theater productions which run each weekend. WYBC Yale Radio is the campus\'s radio station, owned and operated by students. While students used to broadcast on AM & FM frequencies, they now have an Internet-only stream.\n\nTITLE: The Insider\'s Guide to the Colleges\nTEXT: The Insider\'s Guide to the Colleges is a college educational guide which has been published annually by the student editorial staff of the "Yale Daily News" for over four decades. It provides insight to prospective undergraduate students using first-hand accounts of attending students as well as an overview of the admissions process.\n\n"
    - Answer: "1878"
    - Explanation - | 'question': "The Insider's Guide to the Colleges >> author", 'answer': 'Yale Daily News'; 'question': 'In what year was #1 established?', 'answer': '1878' |],

[Inputs: 
    - Question: "Where did Peter and Paul Fortress' designer die?"
    - Context: "TITLE: Peter and Paul Fortress\nTEXT: Today it has been adapted as the central and most important part of the State Museum of Saint Petersburg History. The museum has gradually become virtually the sole owner of the fortress building, except the structure occupied by the Saint Petersburg Mint (Monetniy Dvor).\n\nTITLE: Peter and Paul Fortress\nTEXT: The Peter and Paul Fortress is the original citadel of St. Petersburg, Russia, founded by Peter the Great in 1703 and built to Domenico Trezzini's designs from 1706 to 1740 as a star fortress. In the early 1920s, it was still used as a prison and execution ground by the Bolshevik government.\n\n"
    - Answer: "Saint Petersburg"
    - Explanation - | 'question': 'Who designed Peter and Paul Fortress?', 'answer': 'Domenico Trezzini'; 'question': '#1 >> place of death', 'answer': 'Saint Petersburg' |]


Now answer without explanation

Inputs:

    - Question: "{question}"

    - Context: "{context}"

Carefully read information above and answer the question: {question}
Give me a short answer without explanation. YOUR ANSWER:
"""
        return prompt

    def _build_messages_gen(self, prompt):
        messages = [
            {"role": "system",
             "content": "You are Qwen, a helpful assistant. You need to answer the question briefly."},
            {"role": "user", "content": f"{prompt}"}
        ]
        return messages

    def _generate_gen(self, prompt):
        messages = self._build_messages_gen(prompt)

        text = self.gen_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.gen_tok([text], return_tensors="pt").to(self.gen_model.device)
        outputs = self.gen_model.generate(**model_inputs,
                                          pad_token_id=self.gen_tok.pad_token_id,
                                          eos_token_id=self.gen_tok.eos_token_id,
                                          return_dict_in_generate=True,
                                          output_scores=True,
                                          max_new_tokens=15,
                                          temperature=0.6,
                                          do_sample=False,
                                          top_p=0.95
                                          )
        input_length = model_inputs["input_ids"].shape[1]
        new_tokens = outputs["sequences"][:, input_length:]

        decoded_text = self.gen_tok.decode(new_tokens[0], skip_special_tokens=True)

        return decoded_text


class LLMJudge(AFeedbackModel, GenerationMixin):
    """
    Вызывает внешнюю LLM-оценку в конце эпизода.
    `judge_fn` должен вернуть bool.
    """

    def __init__(
            self,
            gen_model_name: str,
            judge_model_name: str | None = None,
            completion_reward: float = 1.0,
            device_gen: str = "auto",
            device_judge: str = "auto",
    ):
        AFeedbackModel.__init__()
        # всегда загружаем генератор
        GenerationMixin.__init__(self, gen_model_name, device_gen)
        judge_model_name = judge_model_name or gen_model_name

        if judge_model_name == gen_model_name:
            # если имя совпадает — переиспользуем тот же токенизатор и модель
            self.judge_tok = self.gen_tok
            self.judge_model = self.gen_model
        else:
            # иначе — загружаем отдельную judge-модель
            self.judge_tok = AutoTokenizer.from_pretrained(judge_model_name)
            self.judge_model = AutoModelForCausalLM.from_pretrained(
                judge_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device_judge,
                trust_remote_code=True
            )

        self.completion_reward = completion_reward

    def reset(self, obs, info) -> None:
        # completed is always False in this class
        # as it has no stopping mechanism
        self.completed = False

    def _build_prompt_judge(self, question, true_answer, generated_answer):
        check_prompt = f"""QUESTION: {question}

TRUE ANSWER: {true_answer}

GENERATED ANSWER: {generated_answer}

Is GENERATED ANSWER similar to TRUE ANSWER?"""

        return check_prompt

    def _build_messages_judge(self, prompt):

        check_instruction_prompt = """You are a verification system.
You are provided with QUESTION, TRUE ANSWER on this QUESTION and GENERATED ANSWER.
You need to verify is GENERATED ANSWER are similar to TRUE ANSWER in terms of answer to QUESTION.
If you have doubts about GENERATED ANSWER, write "NO", if GENERATED ANSWER is a clear synonym of TRUE ANSWER, write "YES".

You must give your answer in the following format:

Chain of Thoughts
FINAL ANSWER: your final answer (only "YES" or "NO" allowed here) """

        messages = [
            {"role": "system", "content": check_instruction_prompt},
            {"role": "user", "content": f"{prompt}"}
        ]
        return messages

    def _generate_judge(self, prompt):

        messages = self._build_messages_judge(prompt)

        text = self.judge_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.judge_tok([text], return_tensors="pt").to(self.judge_model.device)
        outputs = self.judge_model.generate(**model_inputs,
                                            pad_token_id=self.judge_tok.pad_token_id,
                                            eos_token_id=self.judge_tok.eos_token_id,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_new_tokens=2048,
                                            temperature=0.2,
                                            top_p=0.95
                                            )
        input_length = model_inputs["input_ids"].shape[1]
        new_tokens = outputs["sequences"][:, input_length:]

        decoded_text = self.judge_tok.decode(new_tokens[0], skip_special_tokens=True)

        return decoded_text

    def _judge(self, question: str, passages: List[str], true_answer: str) -> bool:

        prompt_gen = self._build_prompt_gen(passages, question)
        answer_llm = self._generate_gen(prompt_gen)

        print(f'answer_llm - {answer_llm}')
        print(f'true_answer - {true_answer}')

        prompt_judge = self._build_prompt_judge(question, true_answer, answer_llm)
        pred = self._generate_judge(prompt_judge)
        llm_judge = pred.split("FINAL ANSWER: ")[-1]
        print(f'llm_judge - {llm_judge}')
        result = None

        if llm_judge == 'YES':
            result = True
        elif llm_judge == 'NO':
            result = False
        else:
            result = False

        return result

    def reward(self, obs, info, is_final=None) -> float:

        if is_final == True:
            # print(obs["question"], type(obs["question"]))
            # print(obs["retrieved_chunks"], type(obs["retrieved_chunks"]))
            # print(info["answer"], type(info["answer"]))
            llm_as_judge = self._judge(
                obs["question"],
                obs["retrieved_chunks"],
                info["answer"]
            )
            if llm_as_judge:
                reward = self.completion_reward
            else:
                reward = 0.
        else:
            reward = 0.

        return reward


class ExactMatchFeedback(AFeedbackModel, GenerationMixin):
    #ReSearcher, Search-R1
    def __init__(
            self,
            gen_model_name: str,
            completion_reward: float = 1.0,
            device_gen: str = "auto",
    ):
        AFeedbackModel.__init__()
        GenerationMixin.__init__(self, gen_model_name, device_gen) # всегда загружаем генератор
        self.completion_reward = completion_reward

    def reset(self, obs, info) -> None:
        self.completed = False

    def _normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))

    def exact_match(self, true_answer, generated_answer):
        true_answer, generated_answer = self._normalize_answer(true_answer), self._normalize_answer(generated_answer)
        return int(true_answer == generated_answer)

    def reward(self, obs, info, is_final=None) -> float:
        if is_final:
            # print(obs["question"], type(obs["question"]))
            # print(obs["retrieved_chunks"], type(obs["retrieved_chunks"]))
            # print(info["answer"], type(info["answer"]))
            prompt_gen = self._build_prompt_gen(obs["retrieved_chunks"], obs["question"])
            answer_llm = self._generate_gen(prompt_gen)
            print(f"answer_llm - {answer_llm}")
            print(f"true_answer - {info['answer']}")
            em = self.exact_match(
                info["answer"],
                answer_llm
            )
            if em:
                reward = self.completion_reward
            else:
                reward = 0.
        else:
            reward = 0.

        return reward


class MutualInformationFeedback(AFeedbackModel, GenerationMixin):
    #-log p(y|q,F) - log p(y|q)
    def __init__(
            self,
            gen_model_name: str,
            completion_coeff: float = 1.0,
            device_gen: str = "auto",
    ):
        AFeedbackModel.__init__()
        GenerationMixin.__init__(self, gen_model_name, device_gen) # всегда загружаем генератор
        self.completion_coeff = completion_coeff

    def reset(self, obs, info) -> None:
        self.completed = False

    def calculate_nll(self,
                      prompt: str,
                      response: str,
                      ) -> torch.Tensor:  # Возвращает тензор с NLL для каждого токена ответа
        """
        Вычисляет отрицательный логарифм правдоподобия (NLL) для каждого
        токена в 'response' для данного 'prompt'.

        Args:
            prompt (str): Входной промт для модели.
            response (str): Правильный ответ, NLL токенов которого нужно вычислить.
            model: Загруженная модель Hugging Face.
            tokenizer: Загруженный токенизатор Hugging Face.
            device (str, optional): Устройство для вычислений ('cuda', 'cpu', 'mps').

        Returns:
            torch.Tensor: Тензор с NLL для каждого токена в ответе.
                        Размер тензора равен количеству токенов в ответе.
                        Возвращает пустой тензор, если ответ пустой или не удалось
                        вычислить NLL.
        """
        if not response:
            print("Предупреждение: Ответ пустой. Возвращен пустой тензор.")
            return torch.tensor([], device=self.device_gen)

        # print(f"Используемое устройство: {device}")

        if self.gen_tok.pad_token is None:
            if self.gen_tok.eos_token is not None:
                print("Warning: tokenizer.pad_token is None, using tokenizer.eos_token instead.")
                self.gen_tok.pad_token = self.gen_tok.eos_token
            else:
                # Можно установить стандартный паддинг токен, если его совсем нет
                # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # model.resize_token_embeddings(len(tokenizer)) # Важно при добавлении новых токенов
                print("Warning: No pad token found. Using EOS token if available.")
                if self.gen_tok.eos_token is None:
                    raise ValueError("Tokenizer must have a pad token or an eos token.")
                self.gen_tok.pad_token = self.gen_tok.eos_token

        self.gen_model.eval()

        # Токенизация промпта и ответа
        prompt_tokens = self.gen_tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_token_ids = prompt_tokens.input_ids
        prompt_len = prompt_token_ids.shape[1]

        full_text = prompt + response
        inputs = self.gen_tok(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.gen_model.device)

        # ---- Логика определения response_start_index ----
        # (Ваш существующий код для определения response_start_index)
        # Убедимся, что он здесь присутствует и работает
        full_token_ids_list = input_ids[0].tolist()
        prompt_token_ids_list = prompt_tokens.input_ids[0].tolist()
        response_start_index = -1

        # Поиск точного совпадения токенов промпта
        for i in range(len(full_token_ids_list) - len(prompt_token_ids_list) + 1):
            if full_token_ids_list[i:i + len(prompt_token_ids_list)] == prompt_token_ids_list:
                # Проверяем, что после промпта есть еще токены
                if len(full_token_ids_list) > i + len(prompt_token_ids_list):
                    response_start_index = i + len(prompt_token_ids_list)
                    # print(f"Info: Found prompt tokens match. Response starts at index {response_start_index}.")
                    break

        # Запасной вариант: использовать длину токенизированного промпта (может быть неточным из-за BOS)
        if response_start_index == -1:
            prompt_tokens_with_special = self.gen_tok(prompt, return_tensors="pt")
            maybe_prompt_len_in_full = prompt_tokens_with_special.input_ids.shape[1]
            # Простая проверка: если начало совпадает
            if full_token_ids_list[:maybe_prompt_len_in_full] == prompt_tokens_with_special.input_ids[0].tolist() \
                    and len(full_token_ids_list) > maybe_prompt_len_in_full:
                response_start_index = maybe_prompt_len_in_full
                print(f"Info: Using prompt length ({response_start_index}) including potential special tokens.")
            else:
                # Совсем крайний случай - если промпт пустой, ответ начинается с 0 (или 1, если есть BOS)
                if not prompt:
                    # Проверяем, добавляет ли токенизатор BOS токен по умолчанию
                    test_tokenization = self.gen_tok("test", return_tensors='pt').input_ids
                    if test_tokenization.shape[1] > 1 and test_tokenization[0, 0] == self.gen_tok.bos_token_id:
                        response_start_index = 1
                        print("Info: Empty prompt, assuming response starts after BOS token (index 1).")
                    else:
                        response_start_index = 0
                        print("Info: Empty prompt, assuming response starts at index 0.")

                else:
                    print("Error: Couldn't reliably determine the start of the response tokens.")
                    print(f"Prompt tokens (no special): {prompt_token_ids_list}")
                    print(f"Full tokens: {full_token_ids_list}")
                    # Можно попробовать вернуть пустой тензор или вызвать исключение
                    return torch.tensor([], device=self.gen_model.device)  # Возвращаем пустой тензор при ошибке

        # Проверка, что индекс не выходит за границу
        if response_start_index >= len(full_token_ids_list):
            print(
                f"Error: Calculated response start index ({response_start_index}) is out of bounds (sequence length: {len(full_token_ids_list)}).")
            return torch.tensor([], device=self.gen_model.device)
        # ---- Конец логики определения response_start_index ----

        # Создаем `labels`
        labels = input_ids.clone()
        labels[0, :response_start_index] = -100  # Маскируем токены промпта

        # Получаем логиты и вычисляем покомпонентные потери
        with torch.no_grad():
            outputs = self.gen_model(input_ids=input_ids, labels=labels)

            logits = outputs.logits  # Получаем логиты [batch_size, seq_length, vocab_size]

            # Важно: Логиты для позиции i предсказывают токен на позиции i+1.
            # Поэтому нам нужно сдвинуть логиты и метки относительно друг друга.
            # Логиты: берем все до предпоследнего [batch_size, seq_length-1, vocab_size]
            # Метки: берем все со второго [batch_size, seq_length-1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Инициализируем функцию потерь БЕЗ усреднения (reduction='none')
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

            shift_logits_float32 = shift_logits.float()

            # Вычисляем NLL для КАЖДОГО токена в последовательности (где label != -100)
            # Используем преобразованные логиты float32
            # Нужно развернуть batch и seq_len измерения для CrossEntropyLoss
            per_token_nll_all = loss_fct(shift_logits_float32.view(-1, shift_logits_float32.size(-1)),
                                         shift_labels.view(-1))

            # Количество токенов ответа
            num_response_tokens = (shift_labels != -100).sum().item()

            if num_response_tokens == 0:
                print(
                    "Предупреждение: Не найдено токенов ответа для расчета потерь после сдвига. Возможно, ответ слишком короткий или ошибка в индексации.")
                return torch.tensor([], device=self.gen_model.device)

            # Создаем маску для немаскированных токенов в `shift_labels`
            response_token_mask = (shift_labels != -100).view(-1)  # Развернутая маска

            # Выбираем NLL только для токенов ответа с использованием маски
            per_token_nll_response = per_token_nll_all[response_token_mask]

            # Проверка размерности: должна совпадать с количеством токенов ответа
            if per_token_nll_response.shape[0] != num_response_tokens:
                print(
                    f"Warning: Mismatch in expected ({num_response_tokens}) and calculated ({per_token_nll_response.shape[0]}) response token NLLs.")

            # Необязательная проверка: среднее значение должно быть близко к outputs.loss
            if num_response_tokens > 0:
                calculated_avg_nll = per_token_nll_response.mean()
                model_avg_nll = outputs.loss  # loss уже посчитан по немаскированным токенам
                # print(f"Проверка: Средний NLL (расчетный, тип {calculated_avg_nll.dtype}): {calculated_avg_nll.item():.4f}, Средний NLL (модель, тип {model_avg_nll.dtype}): {model_avg_nll.item():.4f}")
                if not torch.isclose(calculated_avg_nll.float(), model_avg_nll.float(), atol=1e-3):
                    print("Предупреждение: Расчетный средний NLL значительно отличается от NLL модели!")

        # Возвращаем тензор с NLL для каждого токена ответа
        return model_avg_nll  # input_ids.cpu(), response_start_index

    def reward(self, obs, info, is_final=None) -> float:
        if is_final:
            # print(obs["question"], type(obs["question"]))
            # print(obs["retrieved_chunks"], type(obs["retrieved_chunks"]))
            # print(info["answer"], type(info["answer"]))
            prompt_gen = self._build_prompt_gen(obs["retrieved_chunks"], obs["question"])
            answer_llm = self._generate_gen(prompt_gen)

            print(f'answer_llm - {answer_llm}')
            print(f"true_answer - {info['answer']}")

            prompt_empty = self._build_prompt_gen([""], obs["question"])
            nll_true_answer = self.calculate_nll(prompt_gen, info["answer"])
            nll_empty = self.calculate_nll(prompt_empty, info["answer"])
            print(f"nll_true_answer - {nll_true_answer}")
            print(f"nll_empty - {nll_empty}")

            reward = -(nll_true_answer - nll_empty)

            reward = reward * self.completion_coeff
        else:
            reward = 0.

        return reward


class StepwiseFactRelevanceFeedback(AFeedbackModel):
    def __init__(self, model_name_сritic: str, per_fact_reward=0.1, completion_reward: float = 1.0,
                 device_critic: str = "auto"):
        self.critic_tok = AutoTokenizer.from_pretrained(model_name_сritic)
        self.critic_model = AutoModelForCausalLM.from_pretrained(
            model_name_сritic,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_critic,
            trust_remote_code=True
        )
        self.per_fact_reward = per_fact_reward
        self.completion_reward = completion_reward
        self.completed = False

    def reset(self, obs, info) -> None:
        self.completed = False

    def _build_prompt_critic(self, val_facts, candt_fact, question, answer):
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
    - Validated Fact: "{val_facts}"
    - Candidate Fact: "{candt_fact}"

YOUR ANSWER:<|im_end|>\n<|im_start|>assistant\n<think>\n'''
        return prompt

    def _generate_yes(self, prompt):
        # Токенизируем входной промт и перемещаем на устройство модели
        inputs = self.critic_tok(prompt, return_tensors="pt", truncation=True).to(self.critic_model.device)

        # Генерируем ответ с включением оценок логитов для каждого шага
        outputs = self.critic_model.generate(
            **inputs,
            pad_token_id=self.critic_tok.pad_token_id,
            eos_token_id=self.critic_tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,  # Обязательно для получения логитов на каждом шаге генерации
            max_new_tokens=35000,
            temperature=0.6,
            top_p=0.95
        )

        # Вычисляем длину исходного ввода и выделяем только сгенерированные токены
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs["sequences"][:, input_length:]

        # Декодируем сгенерированные токены в строку
        answer = self.critic_tok.decode(new_tokens[0], skip_special_tokens=True)

        return answer

    def reward(self, obs, info, is_final=None) -> float:

        # print(obs["question"], type(obs["question"]))
        # print(obs["retrieved_chunks"], type(obs["retrieved_chunks"]))
        # print(info["answer"], type(info["answer"]))

        val_fact = "\n".join(obs["retrieved_chunks"])
        cand_fact = obs['chunks'][info['last_action'][0]]
        question = obs["question"]
        answer = info["answer"]

        prompt = self._build_prompt_critic(val_fact, cand_fact, question, answer)
        pred = self._generate_yes(prompt)
        pred = re.sub(r'.*</think>\s*', '', pred, flags=re.DOTALL)

        if pred == 'Yes':  # or pred =='yes' or pred == 'Yes ' or pred == ' Yes':
            pred = True
        elif pred == 'No':  # or pred =='no' or pred == 'No ' or pred =='no ' or pred == ' No':
            pred = False
        else:
            print(repr(pred))
            pred = False

        if pred:
            reward = self.per_fact_reward
        else:
            reward = 0.

        return reward