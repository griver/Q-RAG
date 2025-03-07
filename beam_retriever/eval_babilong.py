import sys
import os
sys.path.append(os.getcwd()) #fix for importing error

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from torch.utils.data import DataLoader
import logging
torch.random.manual_seed(0)
from tqdm import tqdm
from beam_retriever.retrieval.datasets import collate_fn, BeamRetrieverQAAdapter
from dataloaders.localsets.babilong import RetrievalBabilong
from dataloaders.localsets.musique import RetrievalMusique
from prompts_and_metrics.babilong import (
    DEFAULT_PROMPTS, TEMPLATE, get_formatted_input, compute_exact_match, gen_f1_metric
)

from beam_retriever.retrieval.retriever_model import Retriever
from beam_retriever.utils.utils import load_saved
from dataloaders.globalset import PATHS
import numpy as np
from pathlib import Path
import json
import pandas as pd
from transformers import AutoConfig, AutoModel
import argparse
from beam_retriever.utils.utils import move_to_cuda
from beam_retriever.train_beam_retriever import calc_fact_f1_em, create_dataset
from beam_retriever.eval_qa import init_retriever, init_answerer


def prepare_messages(question, facts, prompt_cfg, user_template):
    str_of_facts = " ".join([f for f in facts])

    #input_text = user_prompt.format(question=question, str_of_facts=str_of_facts, suffix=word_suffix)
    input_text = get_formatted_input(str_of_facts, question, prompt_cfg['examples'],
                                     prompt_cfg['instruction'], prompt_cfg['post_prompt'],
                                     template=user_template)
    #input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(model.device)

    messages = [
        {"role": "system", "content": "Your are an AI assistant, your job is to answer questions given to you by the user."},
        {"role": "user", "content": input_text},
    ]
    return messages


@torch.no_grad()
def evaluate_retriever_and_llm(
        r_tokenizer, retriever, llm_pipe, eval_dataloader,
        compute_f1, compute_exact_match, prompt_cfg, logger, args, log_file
):
    retriever.eval()
    logger.info("begin evaluation")
    df = pd.DataFrame({
        'id': [],
        'question': [],
        'pred': [],
        'target': [],
        'sf_pred': [],
        'sf_target': [],
        'context_len': []
    })

    #all_fact_em, all_fact_f1 = [], []
    all_fact_preds = {}
    all_fact_targets = {}
    all_preds = {}
    all_targets = {}
    all_lens = {}
    all_id = []

    for i, batch in enumerate(tqdm(eval_dataloader)):
        id = batch.pop('id')[0]
        target = batch.pop('answer')[0]
        batch = move_to_cuda(batch)

        fact_pred_beams = retriever(**batch)['current_preds'] #returns N search beams
        fact_pred = fact_pred_beams[0] #select most probable beam
        fact_target = batch['sf_idx'][0]

        fact_tokens = [batch['c_codes'][0][f_id] for f_id in sorted(fact_pred)]
        fact_texts = r_tokenizer.batch_decode(fact_tokens)
        question = r_tokenizer.decode(batch['q_codes'][0], skip_special_tokens=True)

        messages = prepare_messages(question.strip(), fact_texts, prompt_cfg, TEMPLATE)
        output = llm_pipe(messages, **generate_kwargs)
        pred  = output[0]['generated_text']

        all_id.append(id)
        all_fact_preds[id] = fact_pred
        all_fact_targets[id] = fact_target
        all_preds[id] = pred
        all_targets[id] = target
        all_lens[id] = sum(len(c) for c in batch['c_codes'][0]) + len(batch['q_codes'][0])

        df.loc[len(df)] = [id, question, pred, target, str(fact_pred), str(fact_target), all_lens[id]]
        df.to_csv(log_file)

        if (args.num_eval_samples >= 0) and (i >= args.num_eval_samples):
            break

    all_fact_f1, all_fact_em = list(zip(*[calc_fact_f1_em(all_fact_preds[i], all_fact_targets[i]) for i in all_id]))
    # all_fact_em.append(fact_em)
    # all_fact_f1.append(fact_f1)

    fact_em = sum(all_fact_em) / len(all_fact_em)
    fact_f1 = sum(all_fact_f1) / len(all_fact_f1)
    mean_len = np.mean(list(all_lens.values()))

    logger.info(f"Evaluated {len(eval_dataloader)} examples...")
    logger.info(f'Average context len: {mean_len:.2f}')
    logger.info(f"Retriever  Fact EM: {fact_em:.4f}, Fact F1: {fact_f1:.4f}")

    em_scores = [compute_exact_match(all_preds[i], all_targets[i]) for i in all_id]
    f1_scores = [compute_f1(all_preds[i], all_targets[i]) for i in all_id]
    answer_em = np.mean(em_scores)
    answer_f1 = np.mean(f1_scores)
    logger.info(f'Retriever + LLM Answers EM: {answer_em:.4f}, F1: {answer_f1:.4f}')

    return {
        'fact_em':fact_em,
        'fact_f1': fact_f1,
        'mean_len': mean_len,
        'answer_em': answer_em,
        'answer_f1': answer_f1
    }



def eval_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    # model
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base", type=str)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--use_flash_attention", action='store_true')
    parser.add_argument("--flash_attention_type", default='None', type=str)
    #parser.add_argument("--dataset_type", default='hotpot', type=str)
    parser.add_argument("--mean_passage_len", default=120, type=int)
    parser.add_argument("--tokenizer_path", type=str, default='microsoft/deberta-v3-base')
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total sequence length which consists of question and context.")
    parser.add_argument('--use_negative_sampling', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--predict_batch_size", default=1,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument('--gradient_checkpointing', action='store_true')
#    parser.add_argument("--train_file", type=str,
#                        default="data/datasets/mrc/hotpotqa/hotpot_train_v1.1.json")
#    parser.add_argument("--predict_file", type=str,
#                        default="data/datasets/mrc/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--log_period_ratio', type=float, default=0.01)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--warmup-ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--max_eval_batch', default=100, type=int, help='If eval batch is too big split it into chunks of length max_eval_batch')
    parser.add_argument('--num_eval_samples', default=-1, type=int, help='maximum number of samples per evaluation')
    parser.add_argument('--num_chunks', default=50, type=int, help='used only for synthetic datasets where you can control number of samples')
    return parser.parse_args()



if __name__ == "__main__":
    args = eval_args()
    args.dataset = ['babilong',]
    args.babi_task = "qa2"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    args.use_label_order = all([d in ['babilong', 'musique'] for d in args.dataset])

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    verbose = False

    answerer_model_name = "microsoft/Phi-3.5-mini-instruct"
    #answerer_model_name = "microsoft/Phi-4-mini-instruct"
    generate_kwargs = {
        "max_new_tokens": 25,
        "return_full_text": False,
        "temperature": 0.3,
        "do_sample": True,
        # 'num_beams': 1,
        # 'top_p': None,
        # 'top_k': None,
    }

    prompt_cfg = {
        'instruction': DEFAULT_PROMPTS[args.babi_task]['instruction'],
        'examples': DEFAULT_PROMPTS[args.babi_task]['examples'],
        'post_prompt': DEFAULT_PROMPTS[args.babi_task]['post_prompt'],
        'template': TEMPLATE,
    }
    compute_f1 = gen_f1_metric(args.babi_task)

    model, tokenizer = init_answerer(answerer_model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    retriever, r_tokenizer = init_retriever(vars(args))

    log_file = Path(f'./output/babilong_evals/beam-retriever_phi_3.5-mini/{args.num_chunks}_chunks/{args.babi_task}/logs.csv')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    res_file = f'./output/babilong_evals/beam-retriever_phi_3.5-mini/{args.num_chunks}_chunks/{args.babi_task}/results.json'
    cfg_file = f'./output/babilong_evals/beam-retriever_phi_3.5-mini/{args.num_chunks}_chunks/{args.babi_task}/config.json'
    json.dump({'prompt': prompt_cfg, 'generate_kwargs': generate_kwargs}, open(cfg_file, 'w'), indent=4)



    eval_dataset = create_dataset(
        args.dataset, r_tokenizer, "qa2",
        num_chunks=args.num_chunks, seed=args.seed,
        split='eval'
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, pin_memory=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )


    results = evaluate_retriever_and_llm(
        r_tokenizer, retriever, pipe, eval_dataloader,
        compute_f1=compute_f1, compute_exact_match=compute_exact_match,
        prompt_cfg=prompt_cfg, logger=logger, args=args, log_file=log_file
    )
    json.dump(results, open(res_file, 'w'), indent=4)

