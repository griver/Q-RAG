import json
import os
import re
import random
from torch.utils.data import Dataset



'''
{
  "question": "A robe takes 2 bolts of blue fiber and half that much white fiber.
              How many bolts in total does it take?",
  "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount
             of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
}
'''



ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_short_answer(answer) -> str:
    match = ANS_RE.search(answer)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return ""



class RetrievalGSM8K(Dataset):

    def __init__(self, path, split, samples_num, examples_num):
        if split not in ["train", "test"]:
            raise ValueError(f"Unknown split for GSM8K dataset: {split}!")

        if samples_num < 0 or examples_num < 0:
            raise ValueError("samples_num and examples_num must be positive!")

        super().__init__()
        rng = random.Random(100)

        with open(os.path.join(path, "train.jsonl"), 'r', encoding='utf-8') as f:
            train_samples = [json.loads(line) for line in f]
        train_samples_len = len(train_samples)

        if split == "train":
            total_samples = train_samples_len
            if total_samples < samples_num + examples_num:
                raise ValueError("Not enough samples in train dataset!")

            indices = rng.sample(range(total_samples), samples_num + examples_num)
            train_indices = indices[:samples_num]
            example_indices = indices[samples_num:]

            self.samples  = [train_samples[i] for i in train_indices]
            self.examples = [train_samples[i] for i in example_indices]

        elif split == "test":
            # Test dataset -> samples
            with open(os.path.join(path, "test.jsonl"), 'r', encoding='utf-8') as f:
                self.samples = [json.loads(line) for line in f]
            if 0 < samples_num < len(self.samples):
                self.samples = self.samples[:samples_num]

            # Train dataset -> examples
            if 0 < examples_num < train_samples_len:
                example_indices = rng.sample(range(train_samples_len), examples_num)
                self.examples = [train_samples[i] for i in example_indices]
            else:
                self.examples = train_samples

        self._format_examples()
        print(f"GSM8K dataset has been loaded. Samples: {len(self.samples)}, examples: {len(self.examples)}.")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]


    def name(self):
        return "gsm8k"


    def _format_examples(self):
        self.formatted_examples = []
        for example in self.examples:
            if 'question' in example and 'answer' in example:
                self.formatted_examples.append(f"Q: {example['question']}\nA: {example['answer']}")
            else:
                self.formatted_examples.append(str(example))


    def get_examples(self) -> str:
        return self.formatted_examples



if __name__ == '__main__':

    train_dataset = RetrievalGSM8K(path="../Datasets/GSM8K",
        split="train", samples_num=5, examples_num=3)
    print("Name:", train_dataset.name())
    print("Length:", train_dataset.__len__())
    print("Sample 1:", train_dataset[1])
    print("Short answer:", extract_short_answer(train_dataset[1]["answer"]))
    print("\nExamples:"); print(train_dataset.get_examples(), '\n')

