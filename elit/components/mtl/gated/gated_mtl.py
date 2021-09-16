# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-14 18:01
import os
from collections import defaultdict
from typing import Dict, Any, Union, Iterable, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from transformers.models.electra.modeling_electra import ElectraLayer, ElectraSelfAttention, ElectraAttention
from transformers.models.roberta.modeling_roberta import RobertaAttention

from elit.common.dataset import MultiTaskDataLoader
from elit.components.mtl.gated.draw_attn import heatmap
from elit.components.mtl.gated.gated_self_attn import GatedBertSelfAttention, GatedDisentangledSelfAttention, \
    GatedRobertaSelfAttention, GatedElectraSelfAttention
from elit.components.mtl.gated.history import HistoryWithSummary
from elit.components.mtl.multi_task_learning import MultiTaskLearning, MultiTaskModel
from elit.components.mtl.self_teaching import SelfTeaching
from elit.components.mtl.tasks import Task
from elit.layers.embeddings.embedding import Embedding
from elit.layers.transformers.pt_imports import BertModel, AutoModel_
from elit.metrics.mtl import MetricDict
from elit.utils.log_util import cprint
from hanlp_common.util import merge_locals_kwargs
from transformers import BertLayer, optimization
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention
from transformers.models.deberta.modeling_deberta import DebertaAttention, DisentangledSelfAttention, DebertaModel, \
    DebertaLayer


class GatedMultiTaskLearning(MultiTaskLearning):

    def fit(self, encoder: Embedding, tasks: Dict[str, Task], save_dir, epochs, patience=0.5, lr=1e-3, encoder_lr=5e-5,
            adam_epsilon=1e-8, weight_decay=0.0, warmup_steps=0.1, gradient_accumulation=1, grad_norm=5.0,
            encoder_grad_norm=None, decoder_grad_norm=None, tau: float = 0.8, transform=None, eval_trn=True,
            prefetch=None, tasks_need_custom_eval=None, loss_balancer=None, concrete_coef=0.1, encoder_trainable=True,
            self_teaching: Union[int, bool, SelfTeaching] = False, temperature_function=None, kd_loss_function=None,
            freeze_encoder_layers: Optional[Tuple[int, int]] = None, _device_placeholder=False,
            gates_lr=0.001, save_after_each_epoch=False,
            gates_warm_up=None, random=False,
            cache=False, devices=None, logger=None, seed=None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_model(self, training=False, model_cls=MultiTaskModel, **kwargs) -> MultiTaskModel:
        self.model = model = super().build_model(training, model_cls, **kwargs)
        if training and self.config.random:
            cprint('[red]Randomize BERT[/red]')
            model.encoder.transformer = AutoModel_.from_pretrained(self.config.encoder.transformer, training=False)
        for task_name in list(self.tasks):
            if task_name not in self.config.task_names:
                self.config.task_names.append(task_name)
                del self[task_name]
        # noinspection PyTypeChecker
        transformer: Union[BertModel, DebertaModel] = model.encoder.transformer
        self._prune(transformer)
        encoder_trainable = self.config.encoder.trainable
        if not encoder_trainable:
            for k, v in self.model.encoder.named_parameters():
                if k.endswith('.log_a'):
                    v.requires_grad = True
        if self.config.finetune:
            model.decoders.requires_grad_(False)
        return model

    def _prune(self, transformer):
        for layer in transformer.encoder.layer:
            layer: Union[BertLayer, DebertaLayer, ElectraLayer] = layer
            if isinstance(layer.attention, BertAttention):
                # noinspection PyTypeChecker
                gated_attention = GatedBertSelfAttention(transformer.config)
                gated_attention.load_state_dict(layer.attention.self.state_dict(), strict=False)
                layer.attention.self = gated_attention
            elif isinstance(layer.attention, RobertaAttention):
                # noinspection PyTypeChecker
                gated_attention = GatedRobertaSelfAttention(transformer.config)
                gated_attention.load_state_dict(layer.attention.self.state_dict(), strict=False)
                layer.attention.self = gated_attention
            elif isinstance(layer.attention, ElectraAttention):
                # noinspection PyTypeChecker
                gated_attention = GatedElectraSelfAttention(transformer.config)
                gated_attention.load_state_dict(layer.attention.self.state_dict(), strict=False)
                layer.attention.self = gated_attention
            elif isinstance(layer.attention, DebertaAttention):
                # noinspection PyTypeChecker
                gated_attention = GatedDisentangledSelfAttention(transformer.config)
                gated_attention.load_state_dict(layer.attention.self.state_dict(), strict=False)
                layer.attention.self = gated_attention
            else:
                raise NotImplementedError('Unsupported transformer')

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
        reg_loss = self.config.concrete_coef * torch.mean(torch.stack(regs))
        loss = task_loss + reg_loss
        step = history.num_mini_batches
        if writer and step % self.config.gradient_accumulation == 0:
            step //= self.config.gradient_accumulation
            writer.add_scalar('loss/reg', float(reg_loss), step)
            writer.add_scalar(f'loss/{task_name}', float(task_loss), step)
        return loss

    @torch.no_grad()
    def get_gates(self):
        gates = []
        # noinspection PyTypeChecker
        transformer: BertModel = self.model.encoder.transformer
        for layer in transformer.encoder.layer:
            layer: BertLayer = layer
            # noinspection PyTypeChecker
            gated_attention: GatedBertSelfAttention = layer.attention.self
            gates.append(gated_attention.gate.get_gates(False).squeeze())
        return torch.stack(gates)

    def get_num_open_gates(self) -> int:
        return torch.sum((self.get_gates() > 1e-3)).item()

    def get_open_gates_rate(self):
        return torch.mean((self.get_gates() > 1e-3).to(torch.float))

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
                writer.add_scalar(f'{input}/{task_name}', float(scores), global_step=step)
            gates = self.get_gates().cpu().detach().numpy()
            im, cb = heatmap(gates, cbar=True, cmap="binary",
                             row_labels=[f'{x + 1}' for x in range(gates.shape[0])],
                             col_labels=[f'{x + 1}' for x in range(gates.shape[1])],
                             show_axis_labels=True
                             )
            im.set_clim(0, 1)
            plt.xlabel('heads')
            plt.ylabel('layers')
            writer.add_figure('gates', plt.gcf(), global_step=step, close=False)
            plt.clf()
            # save gates
            os.makedirs(f'{save_dir}/gates', exist_ok=True)
            torch.save(gates, f'{save_dir}/gates/{step}.pt')
        return results

    # def _collect_encoder_parameters(self):
    #     if self.config.encoder_trainable:
    #         return super()._collect_encoder_parameters()
    #     # Only train gates
    #     return [v for k, v in self.model.encoder.named_parameters() if k.endswith('.log_a')]

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
            return reports + f' gates: {self.get_open_gates_rate():.2%}'
        return reports

    def prune_heads(self):
        transformer: BertModel = self.model.encoder.transformer
        gates = self.get_gates()
        self._prune(transformer)
        unused_heads = defaultdict(list)
        unused_layers = set()
        gates = gates.tolist()
        for l, layer in enumerate(gates):
            for h, g in enumerate(layer):
                if g < 1e-3:
                    unused_heads[l].append(h)
            if len(unused_heads[l]) == transformer.config.num_attention_heads:
                unused_heads[l].clear()
                unused_layers.add(l)
        print(f'Pruned heads:')
        print(unused_heads)
        print(f'Pruned layers:')
        print(unused_layers)
        transformer.prune_heads(unused_heads)
        transformer.encoder.layer = torch.nn.ModuleList(
            x for i, x in enumerate(transformer.encoder.layer) if i not in unused_layers)
        transformer.to(self.device)
