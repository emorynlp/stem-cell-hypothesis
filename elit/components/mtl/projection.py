# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-27 16:47
import logging
import os
from copy import copy
from typing import Union, Dict, Optional, Tuple, Callable, List
import torch
import torch.nn.functional as F
from torch import nn
from alnlp.modules.util import lengths_to_mask
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader
from elit.layers.transformers.pt_imports import AlbertModel, AutoModel
from transformers.models.albert.modeling_albert import AlbertLayer
from elit.common.cache import SequentialFileCache
from elit.common.dataset import MultiTaskDataLoader
from elit.common.structure import History
from elit.common.torch_component import TorchComponent
from elit.components.mtl.loss_balancer import MovingAverageBalancer
from elit.components.mtl.multi_task_learning import MultiTaskLearning, MultiTaskModel
from elit.components.mtl.self_teaching import SelfTeaching
from elit.components.mtl.tasks import Task
from elit.utils.time_util import CountdownTimer
from elit.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs


def hash_batch(batch):
    return hash(tuple(sum(batch['token'], [])))


class Projector(torch.nn.Module):
    def __init__(self, constructor: Callable, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([constructor() for _ in range(num_layers)])

    def forward(self, hidden_states, mask):
        if isinstance(self.layers[0], nn.LSTM):
            x = pack_sequence(torch.split(hidden_states[mask], mask.sum(dim=-1).tolist()), False)
            x, _ = self.layers[0](x)
            x, _ = pad_packed_sequence(x, True)
            return x
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_index, layer in enumerate(self.layers):
            layer_output = layer(hidden_states, extended_attention_mask)
            hidden_states = layer_output[0]
        return hidden_states

    @property
    def dtype(self) -> torch.dtype:
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype


class MultiTaskModelWithProjector(MultiTaskModel):

    def __init__(self,
                 encoder: torch.nn.Module,
                 scalar_mixes: torch.nn.ModuleDict,
                 decoders: torch.nn.ModuleDict,
                 use_raw_hidden_states: dict,
                 ) -> None:
        super().__init__(encoder, scalar_mixes, decoders, use_raw_hidden_states)
        self.projectors = torch.nn.ModuleDict()


class ProjectedMultiTaskLearning(MultiTaskLearning):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: MultiTaskModelWithProjector = self.model

    # noinspection PyMethodOverriding
    def fit(self,
            source: str,
            targets: Dict[str, str],
            tasks: Dict[str, Task],
            save_dir,
            epochs,
            patience=0.5,
            lr=1e-3,
            num_proj_layers=1,
            proj_layer_type=None,
            gradient_accumulation=1,
            grad_norm=1.0,
            prefetch=None,
            transformer_path=None,
            num_attention_heads=None,
            finetune=False,
            joint_finetune=False,
            _device_placeholder=False,
            cache=False,
            devices=None,
            logger=None,
            seed=None,
            **kwargs):
        trn_data, dev_data, batch_size = 'trn', 'dev', None
        tasks = dict((k, v) for k, v in tasks.items() if k in targets)
        return TorchComponent.fit(
            self, **merge_locals_kwargs(locals(), kwargs, excludes=('self', 'kwargs', '__class__', 'tasks')), **tasks)

    def build_dataloader(self, data, batch_size, shuffle=False, device=None, logger: logging.Logger = None,
                         gradient_accumulation=1, tau: float = 0.8, prune=None, prefetch=None,
                         tasks_need_custom_eval=None, cache=False, debug=False, training=False, self_teaching=0,
                         source: str = None,
                         targets: Dict[str, str] = None,
                         finetune=False,
                         joint_finetune=False,
                         **kwargs) -> DataLoader:
        if shuffle and (not finetune or joint_finetune):  # In training
            config = copy(self.config)
            for each in self.tasks:
                del config[each]
            self.load(source, device, **config)
            self.config.pop('training', None)  # Not sure when this gets saved
            # Remove unused tasks
            for task_name in list(self.tasks.keys()):
                if task_name not in targets:
                    del self[task_name]
            dataloader = MultiTaskDataLoader(training=shuffle, tau=tau)
            for i, (target_name, target_path) in enumerate(targets.items()):
                # Add a new Transformer Layer
                target = MultiTaskLearning()
                target.load(target_path, device)
                # Attach decoder (and other settings in the future)
                self.config[target_name] = target.config[target_name]
                self.tasks[target_name] = target.tasks[target_name]
                self.model.decoders[target_name] = target.model.decoders[target_name]
                # Set training data path
                target[target_name].trn = self[target_name].trn
                target[target_name].sampler_builder = self[target_name].sampler_builder
                logger.info(f'[yellow]{i + 1} / {len(targets)}[/yellow] Building [blue]{data}[/blue] dataset for '
                            f'[cyan]{target_name}[/cyan] ...')
                target_dataloader = target.build_task_dataloader(target_name, target[target_name], 'trn', device,
                                                                 gradient_accumulation, logger, False, debug, True)

                @torch.no_grad()
                def generator():
                    for batch in target_dataloader:
                        h_t, output_dict = target._encode(batch, target_name, None, False, False)
                        mask = batch['mask'] = lengths_to_mask(batch['token_length'])
                        h_s, output_dict = self._encode(batch, target_name, None, False, False, run_projection=False)
                        yield batch, h_s, h_t[mask]

                # noinspection PyTypeChecker
                dataloader.dataloaders[target_name] = SequentialFileCache(
                    generator(), size=len(target_dataloader),
                    filename=f'{cache}/{os.getpid()}-{data}-{target_name.replace("/", "-")}-cache.pt' if isinstance(
                        cache, str) else None)
            return dataloader
        else:
            return super().build_dataloader(data, batch_size, shuffle, device, logger, gradient_accumulation, tau,
                                            prune,
                                            prefetch, tasks_need_custom_eval, cache, debug, training, self_teaching,
                                            **kwargs)

    def build_model(self, training=False, targets: Dict[str, str] = None, num_proj_layers=1, transformer_path=None,
                    num_attention_heads=None, proj_layer_type=None,
                    **kwargs) -> MultiTaskModelWithProjector:
        if training:  # It has been built in build_dataloader
            return self.model
        basic_model: MultiTaskModel = super().build_model(training, **kwargs)
        if transformer_path:
            basic_model.encoder.transformer = AutoModel.from_pretrained(transformer_path)
        model = MultiTaskModelWithProjector(basic_model.encoder, basic_model.scalar_mixes, basic_model.decoders,
                                            basic_model.use_raw_hidden_states)

        def build_proj_layer():
            transformer = basic_model.encoder.transformer
            if proj_layer_type == 'lstm':
                dim = basic_model.encoder.get_output_dim()
                return nn.LSTM(input_size=dim,
                               hidden_size=dim // 2,
                               batch_first=True,
                               bidirectional=True)
            if isinstance(transformer, AlbertModel):
                cls = AlbertLayer
            else:
                raise NotImplementedError(f'We do not know how to create additional layer for {type(transformer)}.')
            config = transformer.config
            if num_attention_heads:
                config = copy(config)
                config.num_attention_heads = 8
            return cls(config)

        for task_name in targets:
            model.projectors[task_name] = Projector(build_proj_layer, num_proj_layers)
        return model

    def build_optimizer(self, trn: MultiTaskDataLoader, epochs, adam_epsilon, weight_decay, warmup_steps, lr,
                        encoder_lr, self_teaching: SelfTeaching = None, finetune=False, joint_finetune=False, **kwargs):
        if finetune or joint_finetune:
            optimizer = torch.optim.Adam(
                list(self.model.projectors.parameters()) + list(self.model.decoders.parameters())
                , lr=lr)
            num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
            # scheduler = LambdaLR(optimizer, lambda current_step: max(
            #     0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps))
            # ))

            num_warmup_steps = int(0.1 * num_training_steps)

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )

            scheduler = LambdaLR(optimizer, lr_lambda)
            return optimizer, scheduler, dict()
        optimizer = dict()
        scheduler = dict()
        for task_name, dataloader in trn.dataloaders.items():
            optimizer[task_name] = torch.optim.Adam(self.model.projectors[task_name].parameters(), lr=lr)
            num_training_steps = len(dataloader) * epochs // self.config.get('gradient_accumulation', 1)
            scheduler[task_name] = LambdaLR(optimizer[task_name], lambda current_step: max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps))
            ))
            # scheduler[task_name] = ExponentialLR(optimizer[task_name], 0.5 ** (1 / num_training_steps))
        return optimizer, scheduler

    def fit_dataloader(self, trn: MultiTaskDataLoader, criterion, optimizer, metric, logger: logging.Logger,
                       history: History, ratio_width=None, gradient_accumulation=1, grad_norm=None, finetune=False,
                       joint_finetune=False,
                       **kwargs):
        if finetune or joint_finetune:
            self.model.encoder.requires_grad_(False)
            if joint_finetune:
                optimizers, schedulers, _ = optimizer
                dataloaders = trn.dataloaders
                total_batches = sum(len(x) for x in dataloaders.values())
                timer = CountdownTimer(total_batches)
                total_loss = 0
                for task_name, dataloader in dataloaders.items():
                    for batch, h_s, h_t in dataloader:
                        mask = batch['mask']
                        h_s = self.model.projectors[task_name](h_s, mask)
                        loss = F.mse_loss(h_s[mask], h_t)
                        output_dict, _ = self.feed_batch(batch, task_name)
                        loss += self.compute_loss(batch, output_dict[task_name]['output'], criterion[task_name],
                                                  self.tasks[task_name])
                        if gradient_accumulation and gradient_accumulation > 1:
                            loss /= gradient_accumulation
                        loss.backward()
                        total_loss += float(loss.item())
                        if history.step(gradient_accumulation):
                            if grad_norm:
                                clip_grad_norm(self.model, grad_norm)
                            optimizers.step()
                            optimizers.zero_grad()
                            schedulers.step()
                        timer.log(f'task: {task_name} loss: {total_loss / (timer.current + 1):.4f}',
                                  ratio_percentage=None, logger=logger)
                return
            else:
                return super().fit_dataloader(trn, criterion, optimizer, metric, logger, history, ratio_width,
                                              gradient_accumulation, **kwargs)
        self.model.train()
        self.model.requires_grad_(False)
        self.model.projectors.requires_grad_(True)
        self.model.encoder.train(False)
        optimizers, schedulers = optimizer
        dataloaders = trn.dataloaders
        total_batches = sum(len(x) for x in dataloaders.values())
        timer = CountdownTimer(total_batches)
        total_loss = 0
        for task_name, dataloader in dataloaders.items():
            for batch, h_s, h_t in dataloader:
                mask = batch['mask']
                h_s = self.model.projectors[task_name](h_s, mask)
                loss = F.mse_loss(h_s[mask], h_t)
                if gradient_accumulation and gradient_accumulation > 1:
                    loss /= gradient_accumulation
                loss.backward()
                total_loss += float(loss.item())
                if history.step(gradient_accumulation):
                    if grad_norm:
                        clip_grad_norm(self.model, grad_norm)
                    optimizers[task_name].step()
                    optimizers[task_name].zero_grad()
                    schedulers[task_name].step()
                timer.log(f'task: {task_name} loss: {total_loss / (timer.current + 1):.4f}',
                          ratio_percentage=None, logger=logger)

    def _encode(self, batch, task_name, output_dict=None, cls_is_bos=False, sep_is_eos=False, run_projection=True):
        h, output_dict = super()._encode(batch, task_name, output_dict, cls_is_bos, sep_is_eos)
        mask = batch.get('mask', None)
        if mask is None:
            mask = lengths_to_mask(batch['token_length'])
        if run_projection:
            h = self.model.projectors[task_name](h, mask)
        return h, output_dict

    def save_weights(self, save_dir, filename='model.pt', trainable_only=False, **kwargs):
        super().save_weights(save_dir, filename, trainable_only, **kwargs)
