# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 13:55
import math

import tensorflow as tf

from elit.common.transform_tf import Transform
from elit.components.taggers.tagger_tf import TaggerComponent
from elit.components.taggers.transformers.transformer_transform_tf import TransformerTransform
from elit.layers.transformers.loader_tf import build_transformer
from elit.layers.transformers.utils_tf import build_adamw_optimizer
from elit.losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropyOverBatchFirstDim
from hanlp_common.util import merge_locals_kwargs


class TransformerTaggingModel(tf.keras.Model):
    def __init__(self, transformer: tf.keras.Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformer

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)


class TransformerTaggerTF(TaggerComponent):
    def __init__(self, transform: TransformerTransform = None) -> None:
        if transform is None:
            transform = TransformerTransform()
        super().__init__(transform)
        self.transform: TransformerTransform = transform

    def build_model(self, transformer, max_seq_length, **kwargs) -> tf.keras.Model:
        model, tokenizer = build_transformer(transformer, max_seq_length, len(self.transform.tag_vocab), tagging=True)
        self.transform.tokenizer = tokenizer
        return model

    def fit(self, trn_data, dev_data, save_dir,
            transformer,
            optimizer='adamw',
            learning_rate=5e-5,
            weight_decay_rate=0,
            epsilon=1e-8,
            clipnorm=1.0,
            warmup_steps_ratio=0,
            use_amp=False,
            max_seq_length=128,
            batch_size=32,
            epochs=3,
            metrics='accuracy',
            run_eagerly=False,
            logger=None,
            verbose=True,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    # noinspection PyMethodOverriding
    def build_optimizer(self, optimizer, learning_rate, epsilon, weight_decay_rate, clipnorm, use_amp, train_steps,
                        warmup_steps, **kwargs):
        if optimizer == 'adamw':
            opt = build_adamw_optimizer(self.config, learning_rate, epsilon, clipnorm, train_steps, use_amp,
                                        warmup_steps, weight_decay_rate)
        else:
            opt = super().build_optimizer(optimizer)
        return opt

    def build_vocab(self, trn_data, logger):
        train_examples = super().build_vocab(trn_data, logger)
        warmup_steps_per_epoch = math.ceil(train_examples * self.config.warmup_steps_ratio / self.config.batch_size)
        self.config.warmup_steps = warmup_steps_per_epoch * self.config.epochs
        return train_examples

    def train_loop(self, trn_data, dev_data, epochs, num_examples, train_steps_per_epoch, dev_steps, model, optimizer,
                   loss, metrics, callbacks, logger, **kwargs):
        history = self.model.fit(trn_data, epochs=epochs, steps_per_epoch=train_steps_per_epoch,
                                 validation_data=dev_data,
                                 callbacks=callbacks,
                                 validation_steps=dev_steps,
                                 # mask out padding labels
                                 # class_weight=dict(
                                 #     (i, 0 if i == 0 else 1) for i in range(len(self.transform.tag_vocab)))
                                 )  # type:tf.keras.callbacks.History
        return history

    def build_loss(self, loss, **kwargs):
        if not loss:
            return SparseCategoricalCrossentropyOverBatchFirstDim()
        return super().build_loss(loss, **kwargs)

    def load_transform(self, save_dir) -> Transform:
        super().load_transform(save_dir)
        self.transform.tokenizer = build_transformer(self.config.transformer, self.config.max_seq_length,
                                                     len(self.transform.tag_vocab), tagging=True, tokenizer_only=True)
        return self.transform
