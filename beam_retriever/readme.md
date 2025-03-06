# Beam Retriever
This repository provides an implementation of the [Beam Retriever](https://arxiv.org/abs/2308.08973) method.
Code inspired by the official implementation, but adapted to datasets and evaluations we are interested in. 
This implementation also provides code to evaluate Beam Retriever in combination with `phi-3.5-mini` model. 

## Overview

Beam Retriever is a retrieval technique that addresses the problem of long-context retrieval for various QA (question answering) tasks. 
This repository contains code to train and evaluate the retriever, both standalone and in combination with the phi-3.5-mini model.

## Training

To start training the Beam Retriever, you can use:

```bash
python3 beam_retriever/train_beam_retriever.py --do_train \
                                               --gradient_checkpointing \
                                               --prefix default_prefix_name \
                                               -d musique \
                                               --train_batch_size 8 \
                                               --learning_rate 1e-5 \
                                               --mean_passage_len 120 \
                                               --fp16 \
                                               --beam_size 2 \
                                               --num_train_epochs 20 \
                                               --warmup-ratio 0.1 \
                                               --accumulate_gradients 4 \
                                               --eval_period_ratio 0.3
```

**Note**: 
- Use the `-d` parameter to specify the dataset. For example, you can set `-d babilong` or `-d hotpotqa` to train on those datasets instead of `musique`.

## Evaluation

### Evaluating on Babilong

To evaluate a pretrained Beam Retriever (together with the phi-3.5-mini model) on **Babilong** with sequences of about 32k tokens, run:

```bash
python3 beam_retriever/eval_babilong.py --init_checkpoint <FOLDER_WITH_PRETRAINED_RETRIEVER_MODEL>/checkpoint_best.pt \
                                        --fp16 \
                                        --beam_size 2 \
                                        --num_chunks 1500
```

- If you wish to handle ~150k tokens, change the `--num_chunks` value to `6000`.

### Evaluating on Musique or HotpotQA

Use the following script for Musique or HotpotQA:

```bash
python3 beam_retriever/eval_qa.py --init_checkpoint <FOLDER_WITH_PRETRAINED_RETRIEVER_MODEL>/checkpoint_best.pt \
                                  -d <DATASET_NAME> \
                                  --fp16 \
                                  --beam_size 2
```

Replace `<DATASET_NAME>` with either `musique` or `hotpotqa` as needed.

---
