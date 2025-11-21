import sys
import os
import math
import torch
from torch import nn
from envs.dataloaders.babilong.retrieval_env import RetrievalPolicy
import transformers
contriever_path = "../contriever"
if contriever_path not in sys.path:
    sys.path.append(contriever_path)
import numpy as np
from src import contriever, dist_utils, utils


import warnings
import logging
logger = logging.getLogger(__name__)

class RetrievalAgent(RetrievalPolicy):

    DEFAULT_OPT = dict(
        optim = 'adamw',
        lr = 5e-05,
        eps = 1e-06,
        weight_decay = 0.01,
        beta1 = 0.9,
        beta2 = 0.98,
        dropout = 0.1,
        scheduler = 'linear',
        total_steps = 500000,
        lr_min_ratio = 0.0,
        warmup_steps = 20000,
    )

    def __init__(self, act_encoder=None, act_tokenizer=None, device="cuda:0", epsilon=0., retrieve_k=1, **opt):
        super().__init__()

        self._set_defaults(opt)
        self.opt = opt
        retriever, tokenizer = self.create_model()
        self.s_tokenizer = tokenizer
        self.s_encoder = retriever.to(device)
        self.device = torch.device(device)
        self.retrieve_k = retrieve_k
        self.epsilon = epsilon
        self.optim, self.scheduler = set_optim(self.s_encoder, **opt)

        if act_encoder is not None:
            self.act_encoder = act_encoder
            self.act_tokenizer = act_tokenizer
        # else:
        #     self.momentum = 0.999
        #     self.act_encoder = copy.deepcopy(retriever)

        #self.optimizer = optimizer

    def create_model(self, model_id="bert-base-uncased", pooling='average', random_init=False):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def _set_defaults(self, opt):
        for k, v in self.DEFAULT_OPT.items():
            opt.setdefault(k, v)

    def update(self, states, actions, rtg, next_states, not_done, verbose=True):
        self.s_encoder.train()

        rtg = torch.as_tensor(rtg, device=self.device, dtype=torch.float).squeeze(-1)
        s_embeds = self.embed(states)
        a_embeds = self.embed(actions, is_states=False)
        loss_fn = nn.MSELoss()
        Q_preds = torch.einsum("bd,bd->b", s_embeds, a_embeds) #B,D x B,D

        train_loss = loss_fn(Q_preds, rtg)
        train_loss.backward()
        self.optim.step()
        self.scheduler.step()
        self.s_encoder.zero_grad()
        return train_loss.cpu().detach().item()

    def embed(self, input, is_states=True):
        if not isinstance(input[0], str):
            input = [" ".join(e) for e in input]


        if is_states:
            batch = self.s_tokenizer(input, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            T = batch.data['input_ids'].size(-1)
            if T > 1024:
                #assert T < 1024, f'Input is too long for Contriever T={T}!'
                warnings.warn(f"Input (len={T}) is exceeding length 1024")
            embeds = self.s_encoder(**batch)
        else:
            with torch.no_grad():
                #may need momentum update here
                if isinstance(input[0], str):
                    batch = self.act_tokenizer(input, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                    embeds = self.act_encoder(**batch)
                else:
                    embeds = torch.as_tensor(input, device=self.device)

        return embeds

    def _momentum_update_act_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def act(self, state):
        self.s_encoder.eval()
        a_mask = state['acts_mask']
        acts_ids = sorted(state['acts_mask'].nonzero()[0])

        if np.random.random() > self.epsilon:
            s = [state['state']]
            assert isinstance(s[0][0],str), "act method expect a single state in form of list of sentences"

            s_embed = self.embed([state['state']])[0]
            a_embed = state['acts_embed'][a_mask].to(self.device)
            scores = a_embed @ s_embed
            #scores = torch.inner(s_embed, a_embed)
            score_ids = torch.argsort(scores, descending=True)
            chosen_ids = [acts_ids[i] for i in score_ids[:self.retrieve_k]]
        else:
            chosen_ids = np.random.choice(acts_ids, size=self.retrieve_k, replace=False)
        return chosen_ids

    def save(self, dir, step):
        checkpoint = {'state_encoder':self.s_encoder.state_dict()}
        if hasattr(self, 'act_encoder'):
            checkpoint['action_encoder'] = self.act_encoder.state_dict()

        path = os.path.join(dir, "checkpoint")
        epoch_path = os.path.join(path, str(step))  # "step-%s" % step)
        os.makedirs(epoch_path, exist_ok=True)
        fp = os.path.join(epoch_path, "checkpoint.pth")
        checkpoint["step"] = step
        checkpoint["optim"] = self.optim.state_dict()
        checkpoint["scheduler"] = self.scheduler.state_dict()
        checkpoint["opt"] = self.opt

        torch.save(checkpoint, fp)
        logger.info(f"Saving model to {epoch_path}")


class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


def set_optim(model, **opt):
    if opt['optim'] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt['lr'],
            betas=(opt['beta1'], opt['beta2']),
            eps=opt['eps'],
            weight_decay=opt['weight_decay']
        )
    else:
        raise NotImplementedError("optimizer class not implemented")

    scheduler_args = {
        "warmup": opt['warmup_steps'],
        "total": opt['total_steps'],
        "ratio": opt['lr_min_ratio'],
    }
    if opt['scheduler'] == "linear":
        scheduler_class = WarmupLinearScheduler
    elif ['opt.scheduler'] == "cosine":
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler

