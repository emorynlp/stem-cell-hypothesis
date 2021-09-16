# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-01 18:41
import logging
import os
from typing import Dict, Union, Optional, Tuple, List, Any

import torch
from toposort import toposort

from elit.common.structure import History
from elit.components.mtl.loss_balancer import MovingAverageBalancer

from elit.utils.log_util import flash
from elit.utils.time_util import CountdownTimer
from elit.utils.torch_util import clip_grad_norm
from hanlp_common.constant import HANLP_VERBOSE
from torch.utils.data import DataLoader

from elit.common.cache import SequentialFileCache
from elit.components.mtl.attn.transformer import JointEncoder
from elit.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs, merge_dict
from elit.components.mtl.attn.attn import TaskAttention
from elit.components.mtl.multi_task_learning import MultiTaskLearning, MultiTaskModel
from elit.components.mtl.self_teaching import SelfTeaching
from elit.components.mtl.tasks import Task
from elit.layers.embeddings.contextual_word_embedding import ContextualWordEmbeddingModule
from elit.layers.embeddings.embedding import Embedding
from elit.layers.transformers.pt_imports import BertModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder


class AttentionSquareModel(MultiTaskModel):
    def __init__(self, encoder: ContextualWordEmbeddingModule, scalar_mixes: torch.nn.ModuleDict,
                 decoders: torch.nn.ModuleDict,
                 use_raw_hidden_states: dict) -> None:
        super().__init__(encoder, scalar_mixes, decoders, use_raw_hidden_states)
        self.adapter: TaskAttention = None


class AttentionSquareMultiTaskLearning(MultiTaskLearning):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: AttentionSquareModel = self.model

    def build_model(self, training=False, model_cls=AttentionSquareModel, finetune: Dict[str, str] = None,
                    reuse_decoder=True,
                    **kwargs) -> MultiTaskModel:
        # noinspection PyTypeChecker
        model: AttentionSquareModel = super().build_model(training, model_cls, **kwargs)
        encoder = model.encoder.transformer = JointEncoder(AutoConfig.from_pretrained(self.config.encoder.transformer))
        decoders = model.decoders
        encoder_size = model.encoder.get_output_dim()
        model.adapter = TaskAttention(encoder_size, self.config.nhead, self.config.num_layers)
        if finetune:
            for task_name in list(finetune.keys()):
                if task_name not in self.config.task_names:
                    finetune.pop(task_name)
            for task_name, path in finetune.items():
                c: MultiTaskLearning = MultiTaskLearning()
                c.load(path, devices=-1)
                # noinspection PyTypeChecker
                transformer: BertModel = c.model.encoder.transformer
                encoder.embeddings = transformer.embeddings
                encoder.encoders[task_name] = transformer.encoder
                self.tasks[task_name] = c.tasks[task_name]
                if reuse_decoder:
                    decoders[task_name] = c.model.decoders[task_name]
                else:
                    task = self.tasks[task_name]
                    decoders[task_name] = task.build_model(encoder_size, training=training, **task.config)
                model.use_raw_hidden_states[task_name] = c.model.use_raw_hidden_states[task_name]
        else:
            encoder.embeddings = BertEmbeddings(encoder.config)
            for task_name in self.tasks:
                encoder.encoders[task_name] = BertEncoder(encoder.config)
        if model.encoder:
            model.encoder.requires_grad_(False)
        return model

    # noinspection PyMethodOverriding
    def fit(self,
            encoder: Embedding, tasks: Dict[str, Task], save_dir, epochs,
            finetune: Dict[str, str],
            patience=0.5, lr=1e-3, encoder_lr=5e-5,
            nhead=4, num_layers=2, reuse_decoder=True,
            adam_epsilon=1e-8, weight_decay=0.0, warmup_steps=0.1, gradient_accumulation=1, grad_norm=5.0,
            encoder_grad_norm=None, decoder_grad_norm=None, tau: float = 0.8, transform=None, eval_trn=True,
            prefetch=None, tasks_need_custom_eval=None, loss_balancer=None,
            self_teaching: Union[int, bool, SelfTeaching] = False, temperature_function=None, kd_loss_function=None,
            freeze_encoder_layers: Optional[Tuple[int, int]] = None, _device_placeholder=False, cache=False,
            devices=None, logger=None, seed=None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def _encode(self, batch, task_name, output_dict: dict = None, cls_is_bos=False, sep_is_eos=False):
        model = self.model
        if output_dict:
            hidden, raw_hidden = output_dict['hidden'], output_dict['raw_hidden']
        else:
            hidden = model.encoder(batch)
            if isinstance(hidden, tuple):
                hidden, raw_hidden = hidden
            else:
                raw_hidden = None
            output_dict = {'hidden': hidden, 'raw_hidden': raw_hidden}

        # Apply transformer to fuse embeddings
        fused_hidden = self.model.adapter(*hidden.values())
        hidden_states = None
        for i, name in enumerate(list(hidden.keys())):
            if name == task_name:
                hidden_states = fused_hidden[:, :, i, :]
                break
        if task_name in model.scalar_mixes:
            scalar_mix = model.scalar_mixes[task_name]
            h = scalar_mix(hidden_states)
        else:
            if model.scalar_mixes:  # If any task enables scalar_mix, hidden_states will be a 4d tensor
                hidden_states = hidden_states[-1, :, :, :]
            h = hidden_states
        # If the task doesn't need cls while h has cls, remove cls
        task = self.tasks[task_name]
        if cls_is_bos and not task.cls_is_bos:
            h = h[:, 1:, :]
        if sep_is_eos and not task.sep_is_eos:
            h = h[:, :-1, :]
        return h, {}

    def _finalize_encoder_parameter_groups(self, parameter_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parameter_groups += [{"params": list(self.model.adapter.parameters()), 'lr': self.config.lr}]
        return parameter_groups

    def build_task_dataloader(self, task_name, task, data, device, gradient_accumulation, logger, cache=False,
                              debug=False, training=False):
        if cache and data in ('trn', 'dev'):
            training = False
            filename = f'{cache}/{data}-{task_name.replace("/", "-")}-cache.pt' if isinstance(
                cache, str) else None
            if filename and os.path.isfile(filename):
                return SequentialFileCache(filename=filename, delete=False, device=device)
        dataloader = super().build_task_dataloader(task_name, task, data, device, gradient_accumulation, logger, False,
                                                   debug, training)
        if cache and data in ('trn', 'dev'):
            @torch.no_grad()
            def generator():
                for batch in dataloader:
                    batch['encoder_output'] = {'hidden': self.model.encoder(batch), 'raw_hidden': None}
                    yield batch

            # noinspection PyUnboundLocalVariable
            dataloader = SequentialFileCache(
                generator(), size=len(dataloader),
                filename=filename, delete=False, device=device)
        return dataloader

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric: MetricDict,
                              save_dir, logger: logging.Logger, devices, patience=0.5, loss_balancer=False,
                              self_teaching=0, cache=None, **kwargs):
        self.model.encoder.to(torch.device('cpu'))
        output = super().execute_training_loop(trn, dev, epochs, criterion, optimizer, metric, save_dir, logger,
                                               devices, patience, loss_balancer, self_teaching, cache, **kwargs)
        return output

    def feed_batch(self, batch: Dict[str, Any], task_name, output_dict=None, run_transform=False, cls_is_bos=False,
                   sep_is_eos=False, results=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        output_dict = batch.get('encoder_output', None)
        return super().feed_batch(batch, task_name, output_dict, run_transform, cls_is_bos, sep_is_eos, results)

    def save_weights(self, save_dir, filename='model.pt', trainable_only=False, **kwargs):
        super().save_weights(save_dir, filename, trainable_only, **kwargs)

    def load(self, save_dir: str, devices=None, verbose=HANLP_VERBOSE, finetune=None, **kwargs):
        if finetune:
            if verbose:
                flash('Building model [blink][yellow]...[/yellow][/blink]')
            self.tasks = dict()
            self.model = self.build_model(
                **merge_dict(self.config, training=False, **kwargs, overwrite=True,
                             inplace=True))
            if verbose:
                flash('')
            self.to(devices)
            self.model.eval()
        else:
            super().load(save_dir, devices, verbose, **kwargs)

    # noinspection PyAttributeOutsideInit
    def on_config_ready(self, **kwargs):
        if self.config.finetune:
            self.config.finetune = None
        else:
            self.tasks = dict((key, task) for key, task in self.config.items() if isinstance(task, Task))
        computation_graph = dict()
        for task_name, task in self.tasks.items():
            dependencies = task.dependencies
            resolved_dependencies = self._resolve_task_name(dependencies)
            computation_graph[task_name] = resolved_dependencies

        # We can cache this order
        tasks_in_topological_order = list(toposort(computation_graph))
        task_topological_order = dict()
        for i, group in enumerate(tasks_in_topological_order):
            for task_name in group:
                task_topological_order[task_name] = i
        self._tasks_in_topological_order = tasks_in_topological_order
        self._task_topological_order = task_topological_order
        self._computation_graph = computation_graph

    def _collect_encoder_parameters(self):
        return []

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, history: History,
                       ratio_width=None, gradient_accumulation=1, encoder_grad_norm=None, decoder_grad_norm=None,
                       patience=0.5, eval_trn=False, loss_balancer: MovingAverageBalancer = None,
                       self_teaching: SelfTeaching = None, temperature_function=None, kd_loss_function=None, **kwargs):
        self.model.train()
        encoder_optimizer, encoder_scheduler, decoder_optimizers = optimizer
        timer = CountdownTimer(len(trn))
        total_loss = 0
        self.reset_metrics(metric)
        model = self.model_
        encoder_parameters = model.encoder.parameters()
        decoder_parameters = model.decoders.parameters()
        for idx, (task_name, batch) in enumerate(trn):
            if 'hidden' not in batch['encoder_output']:
                print(task_name)
                print(batch)
                exit(1)
            decoder_optimizer = decoder_optimizers.get(task_name, None)
            loss = self.compute_loss(batch, self.feed_batch(batch, task_name)[0][task_name]['output'],
                                     criterion[task_name], self.tasks[task_name], task_name, history=history)
            if loss_balancer:
                loss_balancer.append(task_name, float(loss))
                loss *= loss_balancer.weight(task_name)
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += float(loss.item())
            if history.step(gradient_accumulation):
                if self.config.get('grad_norm', None):
                    clip_grad_norm(model, self.config.grad_norm)
                if encoder_grad_norm:
                    torch.nn.utils.clip_grad_norm_(encoder_parameters, encoder_grad_norm)
                if decoder_grad_norm:
                    torch.nn.utils.clip_grad_norm_(decoder_parameters, decoder_grad_norm)
                encoder_optimizer.step()
                encoder_optimizer.zero_grad(set_to_none=True)
                encoder_scheduler.step()
                if decoder_optimizer:
                    if isinstance(decoder_optimizer, tuple):
                        decoder_optimizer, decoder_scheduler = decoder_optimizer
                    else:
                        decoder_scheduler = None
                    decoder_optimizer.step()
                    decoder_optimizer.zero_grad(set_to_none=True)
                    if decoder_scheduler:
                        decoder_scheduler.step()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric if eval_trn else None),
                      ratio_percentage=None,
                      ratio_width=ratio_width,
                      logger=logger)
            del loss
        return total_loss / timer.total
