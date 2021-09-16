# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-15 12:21
from tensorboardX import SummaryWriter

from elit.common.structure import History
from elit.utils.io_util import pushd


class HistoryWithSummary(History):
    def __init__(self, save_dir):
        super().__init__()
        with pushd(save_dir):
            self.writer = SummaryWriter()
