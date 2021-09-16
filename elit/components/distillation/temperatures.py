# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-18 14:19
from abc import ABC, abstractmethod

import torch

from hanlp_common.configurable import AutoConfigurable


def constant_temperature_function(logits_S, logits_T, base_temperature):
    '''
    Remember to detach logits_S
    '''
    return base_temperature


def flsw_temperature_function(beta, gamma, eps=1e-4, *args):
    '''
    adapted from arXiv:1911.07471
    '''

    def flsw_temperature_scheduler(logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        t = logits_T.detach()
        with torch.no_grad():
            v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
            t = t / (torch.norm(t, dim=-1, keepdim=True) + eps)
            w = torch.pow((1 - (v * t).sum(dim=-1)), gamma)
            tau = base_temperature + (w.mean() - w) * beta
        return tau

    return flsw_temperature_scheduler


def cwsm_temperature_function(beta, *args):
    '''
    adapted from arXiv:1911.07471
    '''

    def cwsm_temperature_scheduler(logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        with torch.no_grad():
            v = torch.softmax(v, dim=-1)
            v_max = v.max(dim=-1)[0]
            w = 1 / (v_max + 1e-3)
            tau = base_temperature + (w.mean() - w) * beta
        return tau

    return cwsm_temperature_scheduler


class TemperatureFunction(ABC, AutoConfigurable):

    def __init__(self, base_temperature) -> None:
        super().__init__()
        self.base_temperature = base_temperature

    def __call__(self, logits_S, logits_T):
        return self.forward(logits_S, logits_T)

    @abstractmethod
    def forward(self, logits_S, logits_T):
        raise NotImplementedError()

    @staticmethod
    def from_name(name):
        classes = {
            'constant': ConstantTemperature,
            'flsw': FlswTemperature,
            'cwsm': CwsmTemperature,
        }
        assert name in classes, f'Unsupported temperature function {name}. Expect one from {list(classes.keys())}.'
        return classes[name]()


class FunctionalTemperature(TemperatureFunction):

    def __init__(self, scheduler_func, base_temperature) -> None:
        super().__init__(base_temperature)
        self._scheduler_func = scheduler_func

    def forward(self, logits_S, logits_T):
        return self._scheduler_func(logits_S, logits_T, self.base_temperature)


class ConstantTemperature(TemperatureFunction):
    def forward(self, logits_S, logits_T):
        return self.base_temperature


class FlswTemperature(FunctionalTemperature):
    def __init__(self, beta=1, gamma=1, eps=1e-4, base_temperature=8):
        super().__init__(flsw_temperature_function(beta, gamma, eps), base_temperature)
        self.beta = beta
        self.gamma = gamma
        self.eps = eps


class CwsmTemperature(FunctionalTemperature):
    def __init__(self, beta=1, base_temperature=8):
        super().__init__(cwsm_temperature_function(beta), base_temperature)
        self.beta = beta