# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-04 21:49
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def cdroot():
    """
    cd to project root, so models are saved in the root folder
    """
    os.chdir(root)