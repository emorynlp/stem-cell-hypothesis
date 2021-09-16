# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-14 13:14
from elit.common.structure import ConfigTracker
from elit.utils.statistics.moving_avg import MovingAverage


class MovingAverageBalancer(MovingAverage, ConfigTracker):

    def __init__(self, maxlen=5, intrinsic_weighting=True) -> None:
        super().__init__(maxlen)
        ConfigTracker.__init__(self, locals())
        self.intrinsic_weighting = intrinsic_weighting

    def weight(self, task) -> float:
        avg_losses = dict((k, self.average(k)) for k in self._queue)
        avg_per_task = avg_losses[task]
        if not avg_per_task:
            avg_per_task = 1  # avoid float division by zero
        weight = sum(avg_losses.values()) / avg_per_task
        if self.intrinsic_weighting:
            cur_loss = self._queue[task][-1]
            weight *= cur_loss / avg_per_task

        return weight
