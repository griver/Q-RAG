import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import logging
import nltk
import re

logger = logging.getLogger(__name__)


alphabets= r"([A-Za-z])"
prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = r"[.](com|net|org|io|gov|edu|me)"
digits = r"([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n","<stop>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


class NIAH(Dataset):
    def __init__(self,
                 path,
                 split,
                 seed = 52,
                 length=-1,
                 **kwargs
        ):
        super().__init__()
        # if split not in ['validation', 'test', 'train', 'eval']:
        #     raise ValueError(f'unknown split for NIAH dataset: {split}')
        self.split = split
        np.random.seed(seed)
        self.length = length
        self._load_data(path)

    def name(self):
        return 'niah'

    def _load_data(self, path):
        self.tasks = []
        raw_tasks = []

        if self.split in ['train', 'validation']:
            with open(path + '/validation.jsonl', 'r') as json_file:
                for line in json_file:
                    raw_tasks.append(json.loads(line))

        else:
            print("DATA PATH:", path + f'/{self.split}.jsonl')
            with open(path + f'/{self.split}.jsonl', 'r') as json_file:
                for line in json_file:
                    raw_tasks.append(json.loads(line))

        for task in tqdm(raw_tasks, "NIAH load"):
            self.tasks.append(self._adapt_raw_sample(task))
            
        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]

    def _adapt_raw_sample(self, sample):
        context = sample['input']
        answer = ",".join(sample["outputs"])
        
        # sentences = nltk.sent_tokenize(context)
        sentences = split_into_sentences(context)
        
        question = sentences[-1]
        sf_idx = []

        for idx, s in enumerate(sentences):
            for o in  sample["outputs"]:
                if o in s:
                    sf_idx.append(idx)

        return {
            'id': sample["index"],
            'question': question,
            'answer': answer,
            'chunks': sentences[:-1],
            'sf_idx': sf_idx,
        }


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]