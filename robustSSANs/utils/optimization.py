import math
import os
import pdb
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_utils import SchedulerType


def get_scheduler(
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


class SparseAdamW(Optimizer):
    """
    From InfoBERT:
    upper = 0.9 and lower = 0.5 for the NLI task
    upper = 0.95 and lower = 0.75 for the QA task
    """

    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.1,
            correct_bias: bool = True,
            do_sparse: bool = False,
            upper: float = 95,
            lower: float = 50,
            sparse_first: bool = False,
            sparse_second: bool = False,
            gamma: float = 0.5,
            save_dir: str = None,
            num_warmup_steps: int = 0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = dict()
        self.do_sparse = do_sparse
        self.upper = upper / 100.0
        self.lower = lower / 100.0
        self.reserve_intvl = torch.tensor([self.lower, self.upper]).cuda()
        self.sparse_first = sparse_first
        self.sparse_second = sparse_second
        self.gamma = gamma
        self.save_dir = save_dir
        self.num_warmup_steps = num_warmup_steps

        self.softmax = nn.Softmax(dim=0)

    def check_apply_sparse(self, n):
        if all(term in n for term in ["layer", "weight"]):
            return True
        else:
            return False

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.sparse_second:
            self.calculate_mask(type="ce")

        count = 0
        for group in self.param_groups:
            for n, p in zip(group["params_name"], group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if state["step"] > self.num_warmup_steps and self.do_sparse and self.check_apply_sparse(n):
                    mask = torch.ones_like(grad, dtype=torch.float) * self.gradient_mask["param"][count]
                    count += 1
                else:
                    mask = torch.ones_like(grad, dtype=torch.float)

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                grad.mul_(mask).add_(exp_avg.mul((1.0 - mask)))
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.addcdiv_(exp_avg.mul(mask), denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data.mul(mask), alpha=(-group["lr"] * group["weight_decay"]))

        return loss

    def calculate_mask(self, type):
        params_norm = []
        params_his_norm = []
        steps = None
        gradient_mask = dict()
        for group in self.param_groups:
            for n, p in zip(group["params_name"], group["params"]):
                if not self.check_apply_sparse(n) or p.grad is None:
                    continue
                else:
                    steps = self.state[p]["step"] if len(self.state[p]) else 0
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    params_norm.append(torch.norm(p.grad.data))
                    if type == "ce" and len(self.state[p]):
                        params_his_norm.append(torch.norm(self.state[p]["exp_avg"]))

        if params_norm:
            params_norm = torch.stack(params_norm)
            if type == "kl":
                params_saliency = params_norm
                gradient_mask.update({"param": self.calculate_first_order_param_mask(params_saliency, steps)})
            elif type == "ce":
                params_his_norm = torch.stack(params_his_norm) if len(params_his_norm) else None
                gradient_mask.update(
                    {"param": self.calculate_second_order_param_mask(params_norm, params_his_norm, steps)})

        if self.sparse_first and self.sparse_second and type == "ce":
            joint_mask = (1 - self.gamma) * gradient_mask["param"] + self.gamma * self.gradient_mask["param"]
            self.gradient_mask.update({"param": joint_mask})
        else:
            self.gradient_mask.update(gradient_mask)

    def calculate_first_order_param_mask(self, params_saliency, steps):
        lower, upper = torch.quantile(params_saliency, self.reserve_intvl)
        mask = (params_saliency >= lower) & (params_saliency <= upper)
        return mask.float()

    def calculate_second_order_param_mask(self, params_norm, params_his_norm, steps):
        if params_his_norm is not None:
            params_saliency = torch.abs(0.1 * params_norm.div(params_his_norm) - 1.)
            lower, upper = torch.quantile(params_saliency, self.reserve_intvl)
            mask = (params_saliency >= lower) & (params_saliency <= upper)
        else:
            mask = torch.ones_like(params_norm, dtype=torch.float)
        return mask.float()

    def calculate_first_order_grad_mask(self, param_grad):
        grad_saliency = torch.abs(param_grad)
        lower, upper = torch.quantile(grad_saliency, self.reserve_intvl)
        mask = (grad_saliency >= lower) & (grad_saliency <= upper)
        return mask.float()

    def calculate_second_order_grad_mask(self, param_grad, param_his_grad):
        if param_his_grad is not None:
            grad_saliency = torch.abs(0.1 * param_grad.div(param_his_grad) - 1.)
            lower, upper = torch.quantile(grad_saliency, self.reserve_intvl)
            mask = (grad_saliency >= lower) & (grad_saliency <= upper)
        else:
            mask = torch.ones_like(param_grad, dtype=torch.bool)
        return mask.float()
