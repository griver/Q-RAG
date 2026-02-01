import json
import re
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset



def split_docs(text: str):
    two_parts = text.split("QUESTION:")
    question = two_parts[1].strip()
    docs = two_parts[0]
    documents = re.split(r"Document \d+:", docs)
    documents = [doc.strip() for doc in documents if doc.strip()]
    return question, documents



class RetrievalRulerQA(Dataset):

    def __init__(self, path, length = -1):
        super().__init__()
        self.length = length
        self.samples = []

        with open(path, 'r', encoding='utf-8') as jsonl_file:
            for i, line in enumerate(tqdm(jsonl_file, desc="Load RulerQA")):
                if self.length >= 0 and i >= self.length:
                    break

                sample = json.loads(line)
                question, documents = split_docs(sample["input"])
                sample.update({"question": question, "documents": documents})

                del sample["input"]
                del sample["length_w_model_temp"]
                del sample["answer_prefix"]

                self.samples.append(sample)

        print(f"RulerQA has been loaded. Number of samples: {len(self.samples)}")


    def name(self):
        return "RulerQA"


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]



def convert_to_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    decoder = json.JSONDecoder()
    with open(output_path, 'w', encoding='utf-8') as f_out:
        while data := data.lstrip():  # Removes whitespace and checks if string is empty
            obj, index = decoder.raw_decode(data)
            f_out.write(json.dumps(obj) + '\n')
            data = data[index:]  # Slice the data to start at the next object
    print(f"Successfully converted all samples to {output_path}")



if __name__ == "__main__":

    text = "Document 1:\nContent AAA.\n\nDocument 2:\nContent BBB.\n\nDocument 3:\nContent CCC. QUESTION: some_question"
    question, documents = split_docs(text)
    print(question)
    print(documents, '\n')

    convert_to_jsonl("/mnt/Datasets/Ruler/QA-HotpotQA/4K_100_broken.jsonl", "/mnt/Datasets/Ruler/QA-HotpotQA/4K_100.jsonl")

    d = RetrievalRulerQA("/mnt/Datasets/Ruler/QA-SQuAD/4K_100.jsonl")
    print(d.name(), len(d))
    print("Question:", d[0]['question'])
    print("Number of documents:", len(d[0]['documents']))
    print(d[0]['documents'][0])
    print(d[0]['documents'][29])
