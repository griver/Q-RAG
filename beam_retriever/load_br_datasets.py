from dataloaders.localsets.babilong import RetrievalBabilong
from dataloaders.globalset import GlobalSet, DATASETS, PATHS
from dataloaders.localsets.musique import RetrievalMusique
from beam_retriever.retrieval.datasets import BeamRetrieverQAAdapter
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

    #create_simple = lambda name:
    musique = RetrievalMusique(path=PATHS['musique'], tokenizer=tokenizer, length=-1,
            min_context_len=min_context_filter, max_context_len=max_contex_filter,
            type=type, anno_type=anno_type, seed=seed
    )
    babilong = RetrievalBabilong.create(
        path='data_sources/babilong/', task='qa2', num_chunks=100,
        noise_data_path='pg19-with-sentences/', seed=42
    )

    dataset = BeamRetrieverQAAdapter(
        [babilong,],
        tokenizer, "80:20",
    )
    #train_set = dataset
    #new_set = GlobalSet(datasets, split_strategy="80:20")

    train_set, test_set = dataset.get_train_test_split()
    #new_set.print_statistics() #two much time to wait for generated trajectories

    for i, sample in enumerate(train_set):
        if i > 9:
            break
        print("keys:", list(sample.keys()))
        print('sf_idx:', sample['sf_idx'])
        #print(task.get_prompt("standard_qa"))
        #print("=" * 35)