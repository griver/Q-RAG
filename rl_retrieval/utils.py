import numpy as np
import torch

def all_facts_found(obs, info):
    # returns true when all info['sf_idx'] is found
    chunks_mask = np.asarray(obs['chunks_mask'])
    chosen_idx = set(np.where(chunks_mask == 0)[0])
    sf_idx = set(info['sf_idx'])
    return sf_idx == sf_idx.intersection(chosen_idx)