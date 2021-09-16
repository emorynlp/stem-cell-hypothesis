# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-14 18:01
import os
from typing import Dict, Any, Union, Iterable, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from elit.common.dataset import MultiTaskDataLoader
from elit.components.mtl.gated.draw_attn import heatmap
from elit.components.mtl.gated.gated_self_attn import GatedBertSelfAttention, DictGatedBertSelfAttention
from elit.components.mtl.gated.history import HistoryWithSummary
from elit.components.mtl.multi_task_learning import MultiTaskLearning, MultiTaskModel
from elit.components.mtl.self_teaching import SelfTeaching
from elit.components.mtl.tasks import Task
from elit.layers.embeddings.embedding import Embedding
from elit.layers.transformers.pt_imports import BertModel
from elit.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs
from transformers import BertLayer, optimization
from transformers.models.bert.modeling_bert import BertAttention


class JointGatedMultiTaskLearning(MultiTaskLearning):

    def fit(self, encoder: Embedding, tasks: Dict[str, Task], save_dir, epochs, patience=0.5, lr=1e-3, encoder_lr=5e-5,
            adam_epsilon=1e-8, weight_decay=0.0, warmup_steps=0.1, gradient_accumulation=1, grad_norm=5.0,
            encoder_grad_norm=None, decoder_grad_norm=None, tau: float = 0.8, transform=None, eval_trn=True,
            prefetch=None, tasks_need_custom_eval=None, loss_balancer=None, encoder_trainable=True,
            self_teaching: Union[int, bool, SelfTeaching] = False, temperature_function=None, kd_loss_function=None,
            freeze_encoder_layers: Optional[Tuple[int, int]] = None, _device_placeholder=False,
            gates_lr=1e-3, overlap_loss_coef=0,
            gates_warm_up=None,
            cache=False, devices=None, logger=None, seed=None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_model(self, training=False, model_cls=MultiTaskModel, encoder_trainable=True, **kwargs) -> MultiTaskModel:
        self.model = model = super().build_model(training, model_cls, **kwargs)
        for task_name in list(self.tasks):
            if task_name not in self.config.task_names:
                self.config.task_names.append(task_name)
                del self[task_name]
        # noinspection PyTypeChecker
        transformer: BertModel = model.encoder.transformer
        for layer in transformer.encoder.layer:
            layer: BertLayer = layer
            if isinstance(layer.attention, BertAttention):
                # noinspection PyTypeChecker
                gated_attention = DictGatedBertSelfAttention(transformer.config, task_names=self.tasks)
                gated_attention.load_state_dict(layer.attention.self.state_dict(), strict=False)
                layer.attention.self = gated_attention
            else:
                raise NotImplementedError('Unsupported transformer')
        model.encoder.requires_grad_(encoder_trainable)
        if not encoder_trainable:
            for k, v in self.model.encoder.named_parameters():
                if k.endswith('.log_a'):
                    v.requires_grad = True
        if self.config.finetune:
            model.decoders.requires_grad_(False)
        return model

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion: Callable, task: Task, task_name: str,
                     history: HistoryWithSummary = None) -> torch.FloatTensor:
        task_loss = super().compute_loss(batch, output, criterion, task, task_name, history)
        writer = history.writer if history else None
        if not self.model.training:
            return task_loss
        # noinspection PyTypeChecker
        transformer: BertModel = self.model.encoder.transformer
        regs = []
        for layer in transformer.encoder.layer:
            layer: BertLayer = layer
            # noinspection PyTypeChecker
            gated_attention: GatedBertSelfAttention = layer.attention.self
            regs += gated_attention.cached_regs
            gated_attention.cached_regs.clear()
        # reg_loss = self.tasks[task_name].config.concrete_coef * torch.mean(torch.stack(regs))
        # loss = task_loss + reg_loss
        loss = task_loss
        overlap_loss_coef = self.config['overlap_loss_coef']
        if len(self.tasks) > 1 and overlap_loss_coef:
            current_gates = self.get_gates()
            overlap_loss = []
            for other in self.tasks:
                if other == task_name:
                    continue
                gates = self.get_gates(other)
                overlap_loss.append(current_gates * gates)
            overlap_loss = overlap_loss_coef * torch.stack(overlap_loss).sum()
            loss += overlap_loss
        else:
            overlap_loss = None
        step = history.num_mini_batches
        if writer and step % self.config.gradient_accumulation == 0:
            step //= self.config.gradient_accumulation
            # writer.add_scalar(f'{task_name}/reg', float(reg_loss), step)
            writer.add_scalar(f'{task_name}/task', float(task_loss), step)
            if overlap_loss is not None:
                writer.add_scalar(f'{task_name}/overlap', float(overlap_loss), step)
        return loss

    def get_gates(self, task_name=None):
        gates = []
        # noinspection PyTypeChecker
        transformer: BertModel = self.model.encoder.transformer
        for layer in transformer.encoder.layer:
            layer: BertLayer = layer
            # noinspection PyTypeChecker
            gated_attention: DictGatedBertSelfAttention = layer.attention.self
            gate = gated_attention.gates[task_name] if task_name else gated_attention.gate
            gates.append(gate.get_gates(False).squeeze())
        return torch.stack(gates)

    def get_sparsity_rate(self, task_name=None):
        return torch.mean((self.get_gates(task_name) > 1e-3).to(torch.float))

    def save_weights(self, save_dir, filename='model.pt', trainable_only=False, **kwargs):
        super().save_weights(save_dir, filename, trainable_only, **kwargs)

    def build_history(self, save_dir: str):
        return HistoryWithSummary(save_dir)

    def evaluate_dataloader(self, data: MultiTaskDataLoader, criterion, metric: MetricDict, logger, ratio_width=None,
                            input: str = None, history: HistoryWithSummary = None, save_dir=None, **kwargs):
        results = super().evaluate_dataloader(data, criterion, metric, logger, ratio_width, input, **kwargs)
        writer = history.writer if history else None
        if writer:
            step = history.num_mini_batches // self.config.gradient_accumulation
            for task_name, scores in metric.items():
                writer.add_scalar(f'{task_name}/{input}_score', float(scores), global_step=step)
                gates = self.get_gates(task_name).cpu().detach().numpy()
                im, cb = heatmap(gates, cbar=True, cmap="binary",
                                 row_labels=[f'{x + 1}' for x in range(gates.shape[0])],
                                 col_labels=[f'{x + 1}' for x in range(gates.shape[1])],
                                 show_axis_labels=True
                                 )
                im.set_clim(0, 1)
                plt.xlabel('heads')
                plt.ylabel('layers')
                writer.add_figure(f'{input}_gates/{task_name}', plt.gcf(), global_step=step, close=False)
                # save gates
                folder = f'{save_dir}/gates/{task_name}'
                os.makedirs(folder, exist_ok=True)
                torch.save(gates, f'{folder}/{step}.pt')
                plt.clf()
        return results

    def build_optimizer(self, trn: MultiTaskDataLoader, epochs, adam_epsilon, weight_decay, warmup_steps, lr,
                        encoder_lr, self_teaching: SelfTeaching = None, gates_lr=None,
                        gates_warm_up=None, **kwargs):
        encoder_optimizer, encoder_scheduler, _ = super().build_optimizer(trn, epochs, adam_epsilon,
                                                                          weight_decay, warmup_steps,
                                                                          lr, encoder_lr,
                                                                          self_teaching, **kwargs)
        if not self.config.gates_lr:
            return encoder_optimizer, encoder_optimizer, _
        gates_optimizer = torch.optim.Adam(
            [v for k, v in self.model.encoder.named_parameters() if k.endswith('.log_a')]
            , lr=gates_lr)
        decoder_optimizers = dict((k, gates_optimizer) for k in self.tasks)
        if not gates_warm_up:
            return encoder_optimizer, encoder_scheduler, decoder_optimizers
        num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
        decoder_schedulers = dict((k,
                                   optimization.get_linear_schedule_with_warmup(
                                       decoder_optimizers[k],
                                       num_training_steps * warmup_steps,
                                       num_training_steps)) for k in self.tasks)
        return encoder_optimizer, encoder_scheduler, dict(
            (k, (decoder_optimizers[k], decoder_schedulers[k])) for k in self.tasks)

    def _collect_encoder_parameters(self):
        if self.config.gates_lr:
            return [v for k, v in self.model.encoder.named_parameters() if not k.endswith('.log_a')]
        else:
            return super()._collect_encoder_parameters()

    def report_metrics(self, loss, metrics: MetricDict):
        reports = super().report_metrics(loss, metrics)
        if self.model.training:
            return reports + f' gates: {self.get_sparsity_rate():.2%}'
        return reports

    def _encode(self, batch, task_name, output_dict=None, cls_is_bos=False, sep_is_eos=False):
        output_dict = None
        transformer: BertModel = self.model.encoder.transformer
        for layer in transformer.encoder.layer:
            layer: BertLayer = layer
            if isinstance(layer.attention, BertAttention):
                layer.attention.self.enable_gate(task_name)
        return super()._encode(batch, task_name, output_dict, cls_is_bos, sep_is_eos)
