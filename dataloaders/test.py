from globalset import GlobalSet
from transformers import AutoTokenizer

import json
import os
from ast import literal_eval

tokenizer = AutoTokenizer.from_pretrained("Undi95/Meta-Llama-3-8B-Instruct-hf")
new_set = GlobalSet(["musique", "inf", "loogle", "longb", "novel"], tokenizer, "80:20", type = "any", anno_type = "any")
train_set, test_set = new_set.get_train_test_split()
new_set.print_statistics()

# for i, task in enumerate(new_set):
#     if i > 9:
#         break
#     print(task.question)
#     print(task.context_length)
#     print(task.get_prompt("standard_qa"))
#     print("=" * 35)