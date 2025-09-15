import re
import string
from .answer_metric import AnswerMetric

check_instruction_prompt = """You are a verification system.
You are provided with QUESTION, TRUE ANSWER on this QUESTION and GENERATED ANSWER.
You need to verify is GENERATED ANSWER are similar to TRUE ANSWER in terms of answer to QUESTION.
If you have doubts about GENERATED ANSWER, write "NO", if GENERATED ANSWER is a clear synonym of TRUE ANSWER, write "YES".

You must give your answer in the following format:

Chain of Thoughts
FINAL ANSWER: your final answer (only "YES" or "NO" allowed here)"""

check_prompt = """QUESTION: {question}

TRUE ANSWER: {true_answer}

GENERATED ANSWER: {generated_answer}

Is GENERATED ANSWER similar to TRUE ANSWER? """

qa_instruction_prompt = """You are a question-answer long-context system.
Carefully read all context, pay attention on crucial facts and accurately answer the given question.
Your answer must be a short and direct answer to the QUESTION.
If you need Chain of Thoughts, you can write it, but your answer must be finished with the following template:

Final answer: your final SHORT AND DIRECT answer."""

qa_prompt = """QUESTION:
{question}

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER: """




def normalize_answer(s: str) -> str:
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


class GeneralQAExactMatch(AnswerMetric):

    def __call__(self, prediction, target):
        target, prediction = normalize_answer(target), normalize_answer(prediction)
        return float(target == prediction)


class GeneralQAF1(AnswerMetric):

    @staticmethod
    def recall(prediction, target):
        target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
        len_true = len(target)
        len_good = 0
        for word in prediction:
            if word in target:
                len_good += 1
                target.remove(word)
        return len_good / len_true if len_true > 0 else 1

    @staticmethod
    def precision(prediction, target):
        target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
        len_gen = len(prediction)
        len_good = 0
        for word in target:
            if word in prediction:
                len_good += 1
                prediction.remove(word)
        return len_good / len_gen if len_gen > 0 else 1

    def __call__(self, prediction, target):
        prec = self.precision(prediction, target)
        rec = self.recall(prediction, target)
        if (prec + rec) == 0.:
            return 0.0

        return (2.0 * prec * rec) / (prec + rec)

# def compute_exact_match(prediction, target):
#     target, prediction = normalize_answer(target), normalize_answer(prediction)
#     return int(target == prediction)


# def recall(prediction, target):
#     target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
#     len_true = len(target)
#     len_good = 0
#     for word in prediction:
#         if word in target:
#             len_good += 1
#             target.remove(word)
#     return len_good / len_true if len_true > 0 else 1
#
#
# def precision(prediction, target):
#     target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
#     len_gen = len(prediction)
#     len_good = 0
#     for word in target:
#         if word in prediction:
#             len_good += 1
#             prediction.remove(word)
#     return len_good / len_gen if len_gen > 0 else 1
#
#
# def compute_f1(prediction, target):
#     prec = precision(prediction, target)
#     rec = recall(prediction, target)
#     if (prec + rec) == 0.:
#         return 0.
#
#     f1 = (2. * prec * rec) / (prec + rec)
#     return f1