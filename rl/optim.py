import torch
import math


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
    elif opt['scheduler'] == "cosine":
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler


class LinearAnnealingVal:
    def __init__(self, warmup_steps, max_annealing_steps, start_val, final_val):
        super().__init__()
        self._value = start_val
        self._step = 0
        self.start_val = start_val
        self.final_val = final_val
        self.warmup_steps = warmup_steps
        self.max_annealing_steps = max_annealing_steps

    def step(self):
        return self.set_step(self._step+1)

    def get(self):
        return self._value

    def set_step(self, step):
        self._step = step
        self._value = self.compute_value(self._step)
        return self.get()

    def compute_value(self, step):
        if step <= self.warmup_steps:
            return self.start_val
        elif step > self.max_annealing_steps:
            return self.final_val
        else:
            # Linear interpolation between start_val and end_val
            fraction = (step - self.warmup_steps) / (self.max_annealing_steps - self.warmup_steps)
            val = self.start_val + fraction * (self.final_val - self.start_val)
            return val
