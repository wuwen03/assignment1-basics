import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import math
from collections.abc import Callable, Iterable
from typing import Optional
import numpy.typing as npt
import random
from typing import IO, Any, BinaryIO
import os

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    inputs = inputs.reshape(-1, inputs.size(-1)).to(torch.float32)
    targets = targets.reshape(-1)
    max = inputs.max(-1, keepdim=True).values
    inputs = inputs - max
    sum = inputs.exp().sum(-1)
    batch_loss = -inputs[torch.arange(0, inputs.size(0)),targets] + sum.log()
    return batch_loss.mean()

class SGD(torch.optim.Optimizer):
    """copy from the manual"""
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    """copy from the manual"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * torch.pow(grad, 2)
                lr_t = lr * math.sqrt(1 - math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + \
            0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) *\
            (max_learning_rate - min_learning_rate)
    if it > cosine_cycle_iters:
        return min_learning_rate
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    # parameters = list(parameters)
    # grads = torch.stack(
    #     [
    #         p.grad for p in parameters if p.grad is not None
    #     ]
    # ).reshape(-1)
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return
    grads = torch.cat(grads)
    # torch.nn.utils.clip_grad_norm_()
    l2_norm = (grads * grads).sum().sqrt()
    if l2_norm > max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad = param.grad * max_l2_norm / (l2_norm + eps)

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    train_batch = torch.empty(batch_size, context_length, dtype=torch.long)
    validation_batch = torch.empty(batch_size, context_length, dtype=torch.long)
    for idx in range(batch_size):
        start = random.randint(0, len(dataset) - context_length - 1)
        train_batch[idx] = torch.Tensor(dataset[start:start+context_length])
        validation_batch[idx] = torch.Tensor(dataset[start+1:start+context_length+1])
    train_batch.to(device)
    validation_batch.to(device)
    # print(train_batch, tag_batch)
    return (train_batch, validation_batch)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]