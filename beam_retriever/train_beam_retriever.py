import logging
import os
import sys
sys.path.append(os.getcwd()) #fix for importing error
import random
from datetime import date
import json
import sys
import numpy as np
import torch
import transformers
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)

from retrieval.retriever_model import Retriever
from retrieval.datasets import BeamRetrieverQADataset, collate_fn, BeamRetrieverQAAdapter
from beam_retriever.utils.utils import load_saved, move_to_cuda, AverageMeter
#from retrieval.config import train_args
from sklearn.metrics import f1_score
from dataloaders.localsets.babilong import RetrievalBabilong
from dataloaders.localsets.hotpotqa import RetrievalHotPotQA
from dataloaders.localsets.musique import RetrievalMusique
from dataloaders.globalset import PATHS
import argparse


def create_dataset(datasets_names, tokenizer, task, max_chunk_len=512, num_chunks=50, seed=52, split='train'):

    datasets = []
    for name in datasets_names:
        if name == 'musique':
            d = RetrievalMusique(path=PATHS['musique'], tokenizer=tokenizer, length=-1,
                                   min_context_len=0, max_context_len=1e7,
                                   type='any', anno_type='any', split=split, seed=seed)
        elif name == 'hotpotqa':
            d = RetrievalHotPotQA(path=PATHS['hotpotqa'], tokenizer=tokenizer, length=-1,
                                  min_context_len=0, max_context_len=1e7, seed=seed, split=split)
        elif name == "babilong":
            bl_split = 'test' if split == 'eval' else split #Babi dataset doesn't have eval split
            d = RetrievalBabilong.create(
                path='data_sources/babilong/', task=task, num_chunks=num_chunks,
                noise_data_path='pg19-with-sentences/', seed=seed, split=bl_split
            )
        else:
            raise ValueError(f'{name} is not adapted to Beam Retriever.')

        datasets.append(d)

    dataset = BeamRetrieverQAAdapter(datasets, tokenizer, "80:20", max_chunk_len=max_chunk_len)

    return dataset

def create_dataset_old(tokenizer, file, max_chunk_len, dataset_type):
    dataset = BeamRetrieverQADataset(
         tokenizer, file, max_chunk_len, type=dataset_type
    )
    return dataset


def train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    # model
    parser.add_argument("--model_name", default="model/deberta-v3-base", type=str)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--use_flash_attention", action='store_true')
    parser.add_argument("--flash_attention_type", default='None', type=str)
    #parser.add_argument("--dataset_type", default='hotpot', type=str)
    parser.add_argument("--mean_passage_len", default=70, type=int)
    parser.add_argument("--tokenizer_path", type=str, default='model/deberta-v3-base')
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total sequence length which consists of question and context.")
    parser.add_argument('--use_negative_sampling', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--predict_batch_size", default=1,
                        type=int, help="Total batch size for predictions.")

    # file
    parser.add_argument("-d", "--dataset", type=str, action='append',
                        help="Training datasets. Specify each with a separate flag -d. Available choices: musique, babilong",
                        required=True)
#    parser.add_argument("--train_file", type=str,
#                        default="data/datasets/mrc/hotpotqa/hotpot_train_v1.1.json")
#    parser.add_argument("--predict_file", type=str,
#                        default="data/datasets/mrc/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument("--learning_rate", default=5e-6, type=float, help="learning rate")
    parser.add_argument("--warmupsteps", default=0.1, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--accumulate_gradients", default=1, type=int)
    parser.add_argument("--num_train_epochs", default=12, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--prefix', type=str, default="default_prefix")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--eval_period', type=int, default=-1)
    parser.add_argument('--eval_period_ratio', type=float, default=-1.0)
    parser.add_argument('--log_period_ratio', type=float, default=0.01)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--stop-drop", default=0, type=float)
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--warmup-ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--max_eval_batch', default=100, type=int, help='If eval batch is too big split it into chunks of length max_eval_batch')
    parser.add_argument('--num_eval_samples', default=-1, type=int, help='maximum number of samples per evaluation')
    parser.add_argument('--num_chunks', default=50, type=int, help='used only for synthetic datasets where you can control number of samples')
    return parser.parse_args()


def main():
    args = train_args()
    num_chunks = args.num_chunks
    use_label_order = all([d in ['babilong', 'musique'] for d in args.dataset])  # works only with babilong-qa2 and musique

    transformers.logging.set_verbosity_error()
    if args.fp16:
        # import apex
        # apex.amp.register_half_function(torch, 'einsum')
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-valbsz{args.predict_batch_size}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_path = os.path.join(args.output_dir, "tblogs")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path, exist_ok=True)

    tb_logger = SummaryWriter(tb_path)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bert_config.cls_token_id = tokenizer.cls_token_id
    bert_config.sep_token_id = tokenizer.sep_token_id
    if args.use_flash_attention:
        bert_config.use_memorry_efficient_attention = True
        bert_config.flash_attention_type = args.flash_attention_type
    model = Retriever(bert_config, args.model_name, AutoModel,
                      max_seq_len=args.max_seq_len, mean_passage_len=args.mean_passage_len, beam_size=args.beam_size, use_negative_sampling=args.use_negative_sampling,
                      gradient_checkpointing=args.gradient_checkpointing, use_label_order=use_label_order,
                      max_eval_batch=args.max_eval_batch)


    eval_dataset = create_dataset(args.dataset, tokenizer, "qa2", num_chunks=num_chunks, seed=args.seed, split='eval')
    print(f'EVAL SIZE: {len(eval_dataset)}')
    # eval_dataset = BeamRetrieverQADataset(
    #     tokenizer, args.predict_file, args.max_seq_len, type=args.dataset_type
    # )

    eval_dataloader = DataLoader(
    eval_dataset, batch_size=args.predict_batch_size, pin_memory=True,
    num_workers=args.num_workers, collate_fn=collate_fn)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.do_train and args.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_len, bert_config.max_position_embeddings))

    if args.local_rank == -1 or args.local_rank == 0:
        logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        if args.local_rank == -1 or args.local_rank == 0:
            logger.info(f"begin load trained model from :{args.init_checkpoint}")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)

    if args.local_rank == -1 or args.local_rank == 0:
        logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters,
                         lr=args.learning_rate, eps=args.adam_epsilon)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    

    if args.do_train:
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_f1 = 0
        train_loss_meter = AverageMeter()
        model.train()

        #train_dataset = BeamRetrieverQADataset(tokenizer, args.train_file, args.max_seq_len, type=args.dataset_type)
        train_dataset = create_dataset(args.dataset, tokenizer, "qa2",
            num_chunks=num_chunks, seed=args.seed, split='train')
        print(f'TRAIN SIZE: {len(train_dataset)}')

        if args.local_rank != -1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                        num_workers=args.num_workers, sampler=train_sampler, collate_fn=collate_fn)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                        num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)
    
        
        t_total = len(train_dataloader) // args.accumulate_gradients * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        log_steps = int(len(train_dataloader) // args.accumulate_gradients * args.log_period_ratio)
        eval_steps = int(len(train_dataloader) // args.accumulate_gradients * args.eval_period_ratio)

        if args.local_rank == -1 or args.local_rank == 0:
            logger.info(f'Start training.... log_steps:{log_steps}, eval_steps:{eval_steps}')
        for epoch in range(int(args.num_train_epochs)):
            for batch in tqdm(train_dataloader):
                batch_step += 1
                #print(f'\rbatch_step={batch_step}')
                id = batch.pop('id')
                batch = move_to_cuda(batch)
                if args.fp16:
                    with autocast():
                        loss = model(**batch)['loss']
                else:
                    loss = model(**batch)['loss']
                loss = loss.sum()
                if args.accumulate_gradients > 1:
                    loss = loss / args.accumulate_gradients
                if args.fp16:
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())

                if (batch_step + 1) % args.accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    if args.fp16:
                        # torch.nn.utils.clip_grad_norm_(
                        #     amp.master_params(optimizer), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # torch.nn.utils.clip_grad_norm_(
                        #     model.parameters(), args.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank == -1 or args.local_rank == 0:
                        tb_logger.add_scalar('batch_train_loss',
                                            loss.item(), global_step)
                        tb_logger.add_scalar('smoothed_train_loss',
                                            train_loss_meter.avg, global_step)

                    if global_step % log_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                        logger.info("Step %d Train loss %.8f on epoch=%d, best_metric=%.3f" % (
                        global_step, train_loss_meter.avg, epoch, best_f1))

                    if args.eval_period_ratio > 0 and global_step % eval_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):

                        metric = predict(tokenizer, model, eval_dataloader, logger, args)
                        pred_list = metric['pred_list']
                        metric = metric['em']
                        logger.info("Step %d Train loss %.8f score %.3f on epoch=%d" % (
                        global_step, train_loss_meter.avg, metric, epoch))

                        tb_logger.add_scalar('em',
                                            metric, global_step)
                        if best_f1 < metric:
                            logger.info("Saving model with best score %.3f -> score %.3f on epoch=%d" %
                                        (best_f1, metric, epoch))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            best_f1 = metric
                            json.dump(pred_list, open(os.path.join(args.output_dir, "pred_best.json"), 'w'))

            if args.local_rank == -1 or args.local_rank == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_last.pt"))
                metric = predict(tokenizer, model, eval_dataloader, logger, args)
                pred_list = metric['pred_list']
                metric = metric['em']
                json.dump(pred_list, open(os.path.join(args.output_dir, "pred_last.json"), 'w'))
                logger.info("Step %d Train loss %.8f f1_score %.8f on epoch=%d" % (
                    global_step, train_loss_meter.avg, metric, epoch))

                tb_logger.add_scalar('em',
                                            metric, global_step)
                
                if best_f1 < metric:
                    logger.info("Saving model with best score %.3f -> score %.3f on epoch=%d" %
                                (best_f1, metric, epoch))
                    torch.save(model.state_dict(), os.path.join(
                        args.output_dir, f"checkpoint_best.pt"))
                    best_f1 = metric
                    json.dump(pred_list, open(os.path.join(args.output_dir, "pred_best.json"), 'w'))
                

        logger.info("Training finished!")

    elif args.do_predict:
        metric = predict(tokenizer, model, eval_dataloader, logger, args)
        logger.info(f"test performance {metric['em']}")
        json.dump(metric['pred_list'], open(os.path.join(args.output_dir, "pred.json"), 'w'))

def calc_fact_f1_em(predicted_support_idxs, gold_support_idxs):
    # Taken from hotpot_eval
    cur_sp_pred = set(map(int, predicted_support_idxs))
    gold_sp_pred = set(map(int, gold_support_idxs))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not cur_sp_pred and not gold_sp_pred:
        f1, em = 1.0, 1.0
        f1, em = 1.0, 1.0
    return f1, em

def predict(tokenizer, model, eval_dataloader, logger, args):
    model.eval()
    logger.info("begin evaluation")
    em_tot, f1_tot = [], []
    mean_len = []
    pred_list = {}
    for i, batch in enumerate(tqdm(eval_dataloader)):
        id = batch.pop('id')

        mean_len.append(sum(len(c) for c in batch['c_codes'][0]) + len(batch['q_codes'][0]))
        batch = move_to_cuda(batch)
        with torch.no_grad():
            current_preds = model(**batch)['current_preds']
        pred_list[id[0]] = current_preds[0]
        f1, em = calc_fact_f1_em(current_preds[0], batch['sf_idx'][0])
        em_tot.append(em)
        f1_tot.append(f1)

        if (args.num_eval_samples >= 0) and (i >= args.num_eval_samples):
            break

    em = sum(em_tot) / len(em_tot)
    f1 = sum(f1_tot) / len(f1_tot)
    logger.info(f"evaluated {len(eval_dataloader)} examples...")
    logger.info(f"performance: em: {em}, f1: {f1}")
    logger.info(f'mean length: {np.mean(mean_len)}')
    model.train()
    return {'em':em, 'f1': f1, 'pred_list': pred_list}


def predict_2(tokenizer, model, eval_dataloader, logger, args):
    model.eval()
    logger.info("begin evaluation")
    em_tot, f1_tot = [], []
    pred_list = {}
    for i, batch in enumerate(tqdm(eval_dataloader)):
        id = batch.pop('id')
        batch = move_to_cuda(batch)
        with torch.no_grad():
            current_preds = model(**batch)['current_preds']
        pred_list[id[0]] = current_preds[0]
        f1, em = calc_fact_f1_em(current_preds[0], batch['sf_idx'][0])
        em_tot.append(em)
        f1_tot.append(f1)

    em = sum(em_tot) / len(em_tot)
    f1 = sum(f1_tot) / len(f1_tot)
    logger.info(f"evaluated {len(eval_dataloader)} examples...")
    logger.info(f"performance: em: {em}, f1: {f1}")
    model.train()
    return {'em': em, 'f1': f1, 'pred_list': pred_list}


if __name__ == "__main__":
    main()