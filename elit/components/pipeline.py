# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 00:22
import types
from typing import Callable, Union, Iterable
from elit.components.lambda_wrapper import LambdaComponent
from elit.common.component import Component
from hanlp_common.document import Document
from elit.utils.component_util import load_from_meta
from hanlp_common.io import save_json, load_json
from hanlp_common.reflection import str_to_type, classpath_of
import elit


class Pipe(Component):

    def __init__(self, component: Component, input_key: str = None, output_key: str = None, **kwargs) -> None:
        super().__init__()
        if not hasattr(self, 'config'):
            self.config = {'classpath': classpath_of(self)}
        self.output_key = output_key
        self.input_key = input_key
        self.component = component
        self.kwargs = kwargs
        self.config.update({
            'component': component.config,
            'input_key': self.input_key,
            'output_key': self.output_key,
            'kwargs': self.kwargs
        })

    # noinspection PyShadowingBuiltins
    def predict(self, doc: Document, **kwargs) -> Document:

        unpack = False
        if self.input_key:
            if isinstance(self.input_key, (tuple, list)):
                if isinstance(self.component, LambdaComponent):  # assume functions take multiple arguments
                    input = [doc[key] for key in self.input_key]
                    unpack = True
                else:
                    input = list(list(zip(*sent)) for sent in zip(*[doc[key] for key in self.input_key]))
            else:
                input = doc[self.input_key]
        else:
            input = doc

        if self.kwargs:
            kwargs.update(self.kwargs)
        if unpack:
            kwargs['_hanlp_unpack'] = True
        output = self.component(input, **kwargs)
        if isinstance(output, types.GeneratorType):
            output = list(output)
        if self.output_key:
            if not isinstance(doc, Document):
                doc = Document()
            if isinstance(self.output_key, tuple):
                for key, value in zip(self.output_key, output):
                    doc[key] = value
            else:
                doc[self.output_key] = output
            return doc
        return output

    def __repr__(self):
        return f'{self.input_key}->{self.component.__class__.__name__}->{self.output_key}'

    @staticmethod
    def from_config(meta: dict, **kwargs):
        cls = str_to_type(meta['classpath'])
        component = load_from_meta(meta['component'])
        return cls(component, meta['input_key'], meta['output_key'], **meta['kwargs'])


class Pipeline(Component, list):
    def __init__(self, *pipes: Pipe) -> None:
        super().__init__()
        if not hasattr(self, 'config'):
            self.config = {'classpath': classpath_of(self)}
        if pipes:
            self.extend(pipes)

    def append(self, component: Callable, input_key: Union[str, Iterable[str]] = None,
               output_key: Union[str, Iterable[str]] = None, **kwargs):
        self.insert(len(self), component, input_key, output_key, **kwargs)
        return self

    def insert(self, index: int, component: Callable, input_key: Union[str, Iterable[str]] = None,
               output_key: Union[str, Iterable[str]] = None,
               **kwargs):
        if not input_key and len(self):
            input_key = self[-1].output_key
        if not isinstance(component, Component):
            component = LambdaComponent(component)
        super().insert(index, Pipe(component, input_key, output_key, **kwargs))
        return self

    def __call__(self, doc: Document, **kwargs) -> Document:
        for component in self:
            doc = component(doc)
        return doc

    @property
    def meta(self):
        return {
            'classpath': classpath_of(self),
            'hanlp_version': elit.version.__version__,
            'pipes': [pipe.config for pipe in self]
        }

    @meta.setter
    def meta(self, value):
        pass

    def save(self, filepath):
        save_json(self.meta, filepath)

    def load(self, filepath):
        meta = load_json(filepath)
        self.clear()
        self.extend(Pipeline.from_config(meta))

    @staticmethod
    def from_config(meta: Union[dict, str], **kwargs):
        if isinstance(meta, str):
            meta = load_json(meta)
        return Pipeline(*[load_from_meta(pipe) for pipe in meta['pipes']])
