# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-25 19:48
import math
from typing import Iterable, Tuple, Callable

import torch
from torch.optim import Optimizer
from transformers import optimization

from elit.common.dataset import MultiTaskDataLoader
from elit.components.mtl.gated.joint_gated_mtl import JointGatedMultiTaskLearning
from elit.components.mtl.multi_task_learning import MultiTaskLearning, MultiTaskModel
from elit.components.mtl.self_teaching import SelfTeaching


class FineTuneMultiTaskLearning(MultiTaskLearning):
    def build_model(self, training=False, model_cls=MultiTaskModel, finetune=None, **kwargs) -> MultiTaskModel:
        if finetune:
            mtl = JointGatedMultiTaskLearning()
            mtl.load(finetune, devices=-1)
            del self.config['finetune']
            for task_name, task in self.tasks.items():
                gates = mtl.get_gates(task_name)
                task.gates = gates.detach()

        return super().build_model(training, model_cls, **kwargs)

    def build_optimizer(self, trn: MultiTaskDataLoader, epochs, adam_epsilon, weight_decay, warmup_steps, lr,
                        encoder_lr, self_teaching: SelfTeaching = None, **kwargs):
        encoder_optimizer, encoder_scheduler, decoder_optimizers = super().build_optimizer(trn, epochs, adam_epsilon,
                                                                                           weight_decay, warmup_steps,
                                                                                           lr, encoder_lr,
                                                                                           self_teaching, **kwargs)
        num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
        hidden_size = self.model.encoder.get_output_dim()
        num_heads = self.model.encoder.transformer.config.num_attention_heads
        head_size = hidden_size // num_heads
        attn_params = self._get_encoder_params(attn=True)
        decoder_optimizers = dict()
        for task_name, task in self.tasks.items():
            groups = []
            offset = 0
            num_layers, _ = task.gates.size()
            gates = task.gates.unsqueeze(-1).expand(num_layers, num_heads, head_size).reshape(num_layers, -1).to(
                self.device)
            for layer in range(0, num_layers):
                gate_this_layer = gates[layer]
                for n in range(6):
                    param = attn_params[offset]
                    offset += 1
                    group = {"params": param, 'lr': encoder_lr,
                             'gates': gate_this_layer.unsqueeze(-1) if param.dim() == 2 else gate_this_layer}
                    groups.append(group)
            assert offset == len(attn_params)
            # noinspection PyTypeChecker
            optimizer = AdamW(
                groups,
                lr=encoder_lr,
                weight_decay=weight_decay,
                eps=adam_epsilon,
            )
            scheduler = optimization.get_linear_schedule_with_warmup(optimizer,
                                                                     num_training_steps * warmup_steps,
                                                                     num_training_steps)
            decoder_optimizers[task_name] = (optimizer, scheduler)
        return encoder_optimizer, encoder_scheduler, decoder_optimizers

    def _collect_encoder_parameters(self):
        return self._get_encoder_params(False)

    def _get_encoder_params(self, attn):
        template = ['transformer.encoder.layer.0.attention.self.query.weight',
                    'transformer.encoder.layer.0.attention.self.query.bias',
                    'transformer.encoder.layer.0.attention.self.key.weight',
                    'transformer.encoder.layer.0.attention.self.key.bias',
                    'transformer.encoder.layer.0.attention.self.value.weight',
                    'transformer.encoder.layer.0.attention.self.value.bias']
        attn_names = set()
        for layer in range(12):
            for each in template:
                attn_names.add(each.replace('0', f'{layer}', 1))
        if attn:
            return [p for n, p in self.model.encoder.named_parameters() if n in attn_names]
        else:
            return [p for n, p in self.model.encoder.named_parameters() if n not in attn_names]


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
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

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
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

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                gates = group['gates']
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # p.data.addcdiv_(exp_avg * gates, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
