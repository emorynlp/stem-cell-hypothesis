# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-31 18:02

import torch

from elit.components.mtl.multi_task_learning import MultiTaskModel, MultiTaskLearning


class MaskLayer(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(input_size).uniform_(-1., 1.))

    def forward(self, h):
        mask = torch.sigmoid(self.weight)
        return h * mask


class MaskedMultiTaskModel(MultiTaskModel):
    def __init__(self, encoder: torch.nn.Module, scalar_mixes: torch.nn.ModuleDict, decoders: torch.nn.ModuleDict,
                 use_raw_hidden_states: dict) -> None:
        super().__init__(encoder, scalar_mixes, decoders, use_raw_hidden_states)
        self.mask_layers = torch.nn.ModuleDict()
        for task_name in decoders:
            self.mask_layers[task_name] = MaskLayer(self.encoder.get_output_dim())


class MaskedMultiTaskLearning(MultiTaskLearning):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: MaskedMultiTaskModel = self.model

    def _encode(self, batch, task_name, output_dict=None, cls_is_bos=False, sep_is_eos=False):
        h, output_dict = super()._encode(batch, task_name, output_dict, cls_is_bos, sep_is_eos)
        h = self.model.mask_layers[task_name](h)
        return h, output_dict

    def build_model(self, training=False, model_cls=MaskedMultiTaskModel, **kwargs) -> MultiTaskModel:
        return super().build_model(training, model_cls, **kwargs)

    def _collect_encoder_parameters(self):
        return super()._collect_encoder_parameters() + list(self.model.mask_layers.parameters())
