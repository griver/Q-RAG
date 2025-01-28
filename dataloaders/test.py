from dataloaders.localsets.babilong import LocalSetBabilong
from globalset import GlobalSet, DATASETS, PATHS
from transformers import AutoTokenizer

import json
import os
from ast import literal_eval

if __name__=="__main__":
    #run from LongContext dir
    seed = 52
    min_context_filter=-1
    max_contex_filter=1e7
    # as babilong can actually generate 1e7 samples, you don't want to wait for it
    # to generate 10k samples of length 1e7, especially for training.
    # therefore we need a more realistic range for babilong (e.g. 4k-128k):
    min_gen_context = 4_000 #min length of generated sequences in babilong
    max_gen_context = 128_000 #min length of generated sequences in babilong
    anno_type='any'
    type='any'
    tokenizer = AutoTokenizer.from_pretrained("Undi95/Meta-Llama-3-8B-Instruct-hf")
    proportions="80:20"

    create_simple = lambda name: DATASETS[name](path=PATHS[name], tokenizer=tokenizer, length=-1,
            min_context_len=min_context_filter, max_context_len=max_contex_filter,
            type=type, anno_type=anno_type, seed=seed
        )

    datasets = [create_simple(n) for n in
                ["hotpot"]#, "novel",]
    ]

    # datasets.append(
    #     LocalSetBabilong.create(
    #         PATHS['babilong'], 'qa2', tokenizer,
    #         min_context_len=min_gen_context, max_context_len=max_gen_context,
    #         seed=seed
    #     )
    # )
    new_set = GlobalSet(datasets, split_strategy="80:20")

    train_set, test_set = new_set.get_train_test_split()
    #new_set.print_statistics() #two much time to wait for generated trajectories

    for i, task in enumerate(train_set):
        if i > 9:
            break
        print(task.question)
        print(task.context_length)
        print(task.get_prompt("standard_qa"))
        print("=" * 35)