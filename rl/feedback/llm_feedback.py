from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from typing import Union, Callable
import re, string
import signal
import atexit
import asyncio
from openai import AsyncOpenAI
from rl.feedback.feedback import AFeedbackModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

torch.manual_seed(42)

class LLMGenerator:
    """Incapsulates local llm or api-cliend. Generates new text"""
    ANSWER_DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 32,
        "stop": None,
    }
    ANSWER_DEFAULT_VLLM_CONFIG = {
        "dtype": "bfloat16",
        "quantization": None,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
    }

    def __init__(
            self,
            use_api: bool,
            model_name: str,
            sampling_params: Union[dict, None] = None,
            vllm_config: Union[dict, None] = None,
            api_base_url: str = None,
            api_key: str = '',
            max_at_same_time: int = 20,
            prepare_messages_func: Callable = None,

    ):
        super().__init__()
        self._prepare_messages_func = prepare_messages_func
        if sampling_params:
            sampling_params = {**self.ANSWER_DEFAULT_SAMPLING_PARAMS, **sampling_params}
        else:
            sampling_params = self.ANSWER_DEFAULT_SAMPLING_PARAMS
        self.answer_sampling_params = sampling_params
        self.answer_use_api = use_api
        self.answer_model_name = model_name

        if not use_api:
            if vllm_config:
                vllm_config = {**self.ANSWER_DEFAULT_VLLM_CONFIG, **vllm_config}
            else:
                vllm_config = self.ANSWER_DEFAULT_VLLM_CONFIG
            self.answer_model = LLM(model=model_name, **vllm_config)
            self.answer_tok = AutoTokenizer.from_pretrained(model_name)
            self._answer_shutdown_called = False
            atexit.register(self.__del__)
            signal.signal(signal.SIGINT, lambda *_: (self.__del__(), exit(130)))
            signal.signal(signal.SIGTERM, lambda *_: (self.__del__(), exit(0)))
        else:
            self.answer_api_client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
            self.answer_max_at_same_time = max_at_same_time

    def __del__(self):
        if getattr(self, '_answer_shutdown_called', False):
            return
        self._answer_shutdown_called = True
        if not getattr(self, 'answer_use_api', False):
            if hasattr(self, 'answer_model'):
                del self.answer_model
                self.answer_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _build_answer_messages(self, question: str, pred_chunks: list[str]):
        return self._prepare_messages_func(question, pred_chunks)

    def _generate_answer_local_vllm(self, prompts: list[str]) -> list[str]:
        outputs = self.answer_model.generate(
            prompts=prompts,
            sampling_params=SamplingParams(**self.answer_sampling_params)
        )
        return [out.outputs[0].text.strip() for out in outputs]

    def _generate_answer_api(self, messages_batch: list[list[dict]]) -> list[str]:
        async def _fetch(messages, sem: asyncio.Semaphore):
            async with sem:
                resp = await self.answer_api_client.chat.completions.create(
                    model=self.answer_model_name,
                    messages=messages,
                    **self.answer_sampling_params
                )
                return resp.choices[0].message.content

        async def _gather_all():
            concurrency_limit = min(self.answer_max_at_same_time, len(messages_batch))
            semaphore = asyncio.Semaphore(concurrency_limit)
            tasks = [_fetch(m, semaphore) for m in messages_batch]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(_gather_all(), loop).result()
        except RuntimeError:
            return asyncio.run(_gather_all())

    def _prepare_prompts(self, questions: list[str], pred_chunks_list: list[list[str]]):
        prompts = []
        for q, ch in zip(questions, pred_chunks_list):
            msgs = self._build_answer_messages(q, ch)
            if self.answer_use_api:
                prompts.append(msgs)
            else:
                #maybe chat_template_args should be specified in the __init__
                text = self.answer_tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(text)
        return prompts

    def generate_answers(self, questions: list[str], pred_chunks_list: list[list[str]]) -> list[str]:
        prompts = self._prepare_prompts(questions, pred_chunks_list)
        if self.answer_use_api:
            return self._generate_answer_api(prompts)
        else:
            return self._generate_answer_local_vllm(prompts)

    def generate_answer(self, question: str, pred_chunks: list[str]) -> str:
        return self.generate_answers([question], [pred_chunks])[0]


class AnswerMetricFeedback(AFeedbackModel):
    FEEDBACK_MODEL_NAME = 'EM'

    # ReSearcher, Search-R1
    def __init__(
            self,
            llm_generator: LLMGenerator,
            metric: Callable = None,
            completion_threshold: float = 1.,
            reward_scaling: float = 1.0,
            never_terminate: bool = False,
    ):
        AFeedbackModel.__init__(self, never_terminate=never_terminate)
        self.llm = llm_generator
        self.reward_scaling = reward_scaling
        self.metric = metric
        self.completion_threshold = completion_threshold

    def score_answer_pred(self,
               predicted_answer: Union[str, list[str]],
               true_answer: Union[str, list[str]]
               ) -> Union[float, list[float]]:

        # if intput is not a batch
        if isinstance(predicted_answer, str):
            predicted_answer = [predicted_answer]
            true_answer = [true_answer]

        rewards = []
        for pred_ans, true_ans in zip(predicted_answer, true_answer):
            score = self.metric(pred_ans, true_ans)
            rewards.append(score*self.reward_scaling)

        return rewards[0] if len(rewards) == 1 else rewards

    def reset(self, obs, info):
        super().reset(obs, info)

    def reward(self, obs, info, is_final=None):
        if self.completed: return 0.
        question = obs['question']
        pred_chunks = obs.get('pred_chunks', [])
        predicted_answer = self.llm.generate_answer(question, pred_chunks)
        true_answer = info.get('answer')
        reward = self.score_answer_pred(predicted_answer, true_answer)
        if reward >= self.completion_threshold:
            self.completed = True
        return reward

    def copy(self):
        """
        Создаёт новый экземпляр AnswerMetricFeedback, который ССЫЛАЕТСЯ на тот же llm_generator
        и те же функции/объекты, не создавая новую LLM-модель и не копируя веса.
        """
        return AnswerMetricFeedback(
            llm_generator=self.llm,                 # шарим существующий LLMGenerator
            reward_scaling=self.reward_scaling,
            metric=self.metric,
            never_terminate=self.never_terminate,
        )


#TODO: this class is not working yet

# class LLMJudgeFeedback(AFeedbackModel):
#     """
#     Вызывает внешнюю LLM-оценку в конце эпизода.
#     `judge_fn` должен вернуть bool.
#     """
#     FEEDBACK_MODEL_NAME = 'LLMJudge'
#     DEFAULT_SAMPLING_PARAMS = {
#         "temperature": 0.2,
#         "top_p": 0.95,
#         "max_tokens": 128,
#         "stop": None,
#     }
#     DEFAULT_VLLM_CONFIG = {
#         "dtype": "bfloat16",
#         "quantization": None,
#         "trust_remote_code": True,
#         "tensor_parallel_size": 1,
#         "gpu_memory_utilization": 0.8,
#     }
#
#     def __init__(
#             self,
#             use_api: bool,
#             judge_model_name: str,
#             reward_scaling: float = 1.0,
#
#             sampling_params: Union[dict, None] = None,
#
#             vllm_config: Union[dict, None] = None,
#             api_base_url: str = None,
#             api_key: str = '',
#             max_at_same_time: int = 20,
#     ):
#         AFeedbackModel.__init__(self)
#
#         if sampling_params:
#             sampling_params = {**self.DEFAULT_SAMPLING_PARAMS, **sampling_params}
#         else:
#             sampling_params = self.DEFAULT_SAMPLING_PARAMS
#         self.sampling_params = sampling_params
#         self.use_api = use_api
#         self.reward_scaling = reward_scaling
#         self.judge_model_name = judge_model_name  # сохраняем для API
#
#         if not use_api:
#             if vllm_config:
#                 vllm_config = {**self.DEFAULT_VLLM_CONFIG, **vllm_config}
#             else:
#                 vllm_config = self.DEFAULT_VLLM_CONFIG
#             self.judge_model = LLM(model=judge_model_name, **vllm_config)
#             self.judge_tok = AutoTokenizer.from_pretrained(judge_model_name)
#             self._shutdown_called = False
#
#             # Лаконичная регистрация обработчиков завершения
#             atexit.register(self.__del__)
#             signal.signal(signal.SIGINT, lambda *_: (self.__del__(), exit(130)))
#             signal.signal(signal.SIGTERM, lambda *_: (self.__del__(), exit(0)))
#         else:
#             # если нужен прокси, то переопредели после __init__ атрибут obj.api_client.http_client
#             self.api_client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
#             self.max_at_same_time = max_at_same_time
#
#     def __del__(self):
#         """Корректно завершает работу vLLM (деструктор)"""
#         if getattr(self, '_shutdown_called', False):
#             return
#         self._shutdown_called = True
#
#         if not self.use_api:
#             del self.judge_model
#             self.judge_model = None
#
#             if torch.torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()
#
#     def _build_prompt_judge(self, question: str, predicted_answer: str, true_answer: str) -> str:
#         check_prompt = f"""QUESTION: {question}
#
# TRUE ANSWER: {true_answer}
#
# GENERATED ANSWER: {predicted_answer}
#
# Is GENERATED ANSWER similar to TRUE ANSWER?"""
#
#         return check_prompt
#
#     def _build_messages_judge(self, prompt):
#
#         check_instruction_prompt = """You are a verification system.
# You are provided with QUESTION, TRUE ANSWER on this QUESTION and GENERATED ANSWER.
# You need to verify is GENERATED ANSWER are similar to TRUE ANSWER in terms of answer to QUESTION.
# If you have doubts about GENERATED ANSWER, write "NO", if GENERATED ANSWER is a clear synonym of TRUE ANSWER, write "YES".
#
# You must give your answer in the following format:
#
# Chain of Thoughts
# FINAL ANSWER: your final answer (only "YES" or "NO" allowed here) """
#
#         messages = [
#             {"role": "system", "content": check_instruction_prompt},
#             {"role": "user", "content": f"{prompt}"}
#         ]
#         return messages
#
#     def _generate_judge_local_vllm(self, prompts: list[str]) -> list[str]:
#         all_texts = []
#
#         max_tokens = 0
#         for prompt in prompts:
#             messages = self._build_messages_judge(prompt)
#             text = self.judge_tok.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#             all_texts.append(text)
#             max_tokens = max(max_tokens, len(self.judge_tok.encode(text)))
#         print("max_tokens in judge prompts batch:", max_tokens)
#
#         outputs = self.judge_model.generate(
#             prompts=all_texts,
#             sampling_params=SamplingParams(**self.sampling_params)
#         )
#
#         decoded_texts = [output.outputs[0].text for output in outputs]
#
#         return decoded_texts
#
#     def _generate_judge_api(self, prompts: list[str]) -> list[str]:
#         """Асинхронно обрабатывает список промптов через OpenAI ChatCompletion API."""
#
#         # Встроенная асинхронная корутина для одного вызова
#         async def _fetch(prompt: str, sem: asyncio.Semaphore):
#             async with sem:
#                 messages = self._build_messages_judge(prompt)
#                 resp = await self.api_client.chat.completions.create(
#                     model=self.judge_model_name,
#                     messages=messages,
#                     **self.sampling_params
#                 )
#                 return resp.choices[0].message.content
#
#         async def _gather_all():
#             # Ограничиваем одновременное количество запросов для контроля RPS
#             concurrency_limit = min(self.max_at_same_time, len(prompts))
#             semaphore = asyncio.Semaphore(concurrency_limit)
#             tasks = [_fetch(p, semaphore) for p in prompts]
#             return await asyncio.gather(*tasks)
#
#         # Запускаем корутину, даже если уже существует активный цикл (например, в Jupyter)
#         try:
#             loop = asyncio.get_running_loop()
#             return asyncio.run_coroutine_threadsafe(_gather_all(), loop).result()  # TODO: remove this
#         except RuntimeError:
#             return asyncio.run(_gather_all())
#
#     def _judge(self, questions: list[str], predicted_answers: list[str], true_answers: list[str]) -> list[bool]:
#         """Обрабатывает батч вопросов и возвращает батч решений"""
#         batch_prompts = []
#
#         # Создаем промпты для каждого элемента в батче
#         for question, predicted_answer, true_answer in zip(questions, predicted_answers, true_answers):
#             prompt = self._build_prompt_judge(question, predicted_answer, true_answer)
#             batch_prompts.append(prompt)
#
#         # Генерируем ответы для всего батча
#         if self.use_api:
#             judge_completions = self._generate_judge_api(batch_prompts)
#         else:
#             judge_completions = self._generate_judge_local_vllm(batch_prompts)
#
#         # Обрабатываем каждый ответ
#         decisions = []
#         for completion in judge_completions:
#             judge_decision = completion.split("FINAL ANSWER: ")[-1].strip()
#             decisions.append(judge_decision == 'YES')
#
#         return decisions
#
#     def reward(self,
#                question: Union[str, list[str]],
#                predicted_answer: Union[str, list[str]],
#                true_answer: Union[str, list[str]]
#                ) -> Union[float, list[float]]:
#
#         # Преобразуем в списки если переданы строки
#         if isinstance(question, str):
#             question = [question]
#             predicted_answer = [predicted_answer]
#             true_answer = [true_answer]
#
#         # Получаем батч решений от судьи
#         llm_judgments = self._judge(question, predicted_answer, true_answer)
#
#         # Вычисляем награды для каждого элемента батча
#         rewards = [self.reward_scaling if judgment else 0. for judgment in llm_judgments]
#
#         # Возвращаем одно значение если был передан один элемент
#         return rewards[0] if len(rewards) == 1 else rewards
#
# # TODO: check if this function is a copy of the one in prompts_and_metrics/
# def compute_f1(self, prediction: str, ground_truth: str) -> float:
#     """
#     Compute precision, recall, and F1 score between two strings.
#     Tokenization is done by splitting on non-word characters and lowercasing.
#     """
#     # Tokenize
#     pred_tokens = re.findall(r"\w+", prediction.lower())
#     gt_tokens = re.findall(r"\w+", ground_truth.lower())
#
#     # Count token frequencies
#     pred_counts = Counter(pred_tokens)
#     gt_counts = Counter(gt_tokens)
#
#     # Compute overlap (multiset intersection)
#     overlap = sum(min(pred_counts[token], gt_counts[token]) for token in pred_counts)
#
#     # Precision and Recall
#     precision = overlap / sum(pred_counts.values()) if pred_counts else 0.0
#     recall = overlap / sum(gt_counts.values()) if gt_counts else 0.0
#
#     # F1 score
#     if precision + recall > 0:
#         return 2 * precision * recall / (precision + recall)
#     else:
#         return 0.0


if __name__ == "__main__":
    model = "Qwen/Qwen3-1.7B"

    # Одиночные примеры
    question = "What is the capital of France?"
    predicted_answer = "Paris \n"
    true_answer = "Paris"

    # Батч примеров
    questions_batch = [
                          "What is the capital of France?",
                          "What is the capital of Germany?",
                          "What is 22+22?"
                      ] * 100
    predicted_answers_batch = [
                                  "Pariss",
                                  "Berlin",
                                  "45"
                              ] * 100
    true_answers_batch = [
                             "Paris",
                             "Berlin",
                             "44"
                         ] * 100

    print("Инициализация LLMJudge с vLLM движком...")
    judge_feedback = LLMJudgeFeedback(
        use_api=False,
        judge_model_name=model,
        reward_scaling=3.0
    )

    EM_feedback = ExactMatchFeedback(reward_scaling=3.0)
    F1_feedback = F1ScoreFeedback(reward_scaling=3.0)

    print("----------------Exact Match (одиночный пример)-----------------")
    em_reward = EM_feedback.reward(predicted_answer, true_answer)
    print(f"em_reward: {em_reward}")

    print("----------------Exact Match (батч)-----------------")
    em_rewards_batch = EM_feedback.reward(predicted_answers_batch, true_answers_batch)
    print(f"em_rewards_batch: {em_rewards_batch}")

    print("----------------F1 Score (одиночный пример)-----------------")
    f1_reward = F1_feedback.reward(predicted_answer, true_answer)
    print(f"f1_reward: {f1_reward}")

    print("----------------F1 Score (батч)-----------------")
    f1_rewards_batch = F1_feedback.reward(predicted_answers_batch, true_answers_batch)
    print(f"f1_rewards_batch: {f1_rewards_batch}")

    print("--------------Judge Feedback (одиночный пример)----------------")
    judge_reward = judge_feedback.reward(question, predicted_answer, true_answer)
    print(f"judge_reward: {judge_reward}")

    print("--------------Judge Feedback (батч)----------------")
    judge_rewards_batch = judge_feedback.reward(questions_batch, predicted_answers_batch, true_answers_batch)
    print(f"judge_rewards_batch: {judge_rewards_batch}")

    # judge_feedback_api = LLMJudgeFeedback(
    #     use_api=True,
    #     judge_model_name="/trinity/home/i.evdokimov/models/Qwen2.5-7B-Instruct",
    #     reward_scaling=3.0,
    #     api_key="some-key",
    #     api_base_url='http://localhost:10001/v1',
    #     max_at_same_time=25,
    # )

    # print("--------------Judge Feedback with API (одиночный пример)----------------")
    # judge_reward_api = judge_feedback_api.reward(question, predicted_answer, true_answer)
    # print(f"API judge_reward: {judge_reward_api}")

    # print("--------------Judge Feedback with API (батч)----------------")
    # judge_rewards_batch_api = judge_feedback_api.reward(questions_batch, predicted_answers_batch, true_answers_batch)
    # print(f"API judge_rewards_batch: {judge_rewards_batch_api}")