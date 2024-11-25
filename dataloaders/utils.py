from typing import List
from dataclasses import dataclass, field

TEMPLATES = {
    "standard_qa": {
        "args": ["context", "question"],
        "text": """There is a context which is crucial to answer question '{question}':
{context}

Carefully read information above and answer the question: {question}
YOUR ANSWER: """
    },

    "standard_summary": {
        "args": ["context"],
        "text": """There is a context you need to summarize into comprehensive and short paragraphs:
{context}

Carefully read information above and summarize it into one, two or three paragraphs of text. 
This summary must be expressive and accurate.
YOUR SUMMARY: """
    },
}

@dataclass
class Task:
    type: str  # summary or qa
    anno_type: str  # synth or real
    context_length: int
    context: str
    answer: str
    question: str
    dataset_name: str
    dataset_partition: str = ""
    options: List = field(default_factory=lambda: [])

    def get_prompt(self, template = None, need_answer = False):
        assert template in TEMPLATES
        args = {name: self.__getattribute__(name) for name in TEMPLATES[template]["args"]}
        prompt = TEMPLATES[template]["text"].format(**args)
        if need_answer:
            prompt += self.answer
        return prompt