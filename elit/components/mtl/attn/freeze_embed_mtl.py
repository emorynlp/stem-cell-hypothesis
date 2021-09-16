# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-01 16:20
from elit.components.mtl.multi_task_learning import MultiTaskLearning, MultiTaskModel


class FreezeEmbeddingMultiTaskLearning(MultiTaskLearning):
    def build_model(self, training=False, model_cls=MultiTaskModel, **kwargs) -> MultiTaskModel:
        model = super().build_model(True, model_cls, **kwargs)
        model.encoder.transformer.embeddings.requires_grad_(False)
        return model

    def save_weights(self, save_dir, filename='model.pt', trainable_only=False, **kwargs):
        super().save_weights(save_dir, filename, trainable_only, **kwargs)
