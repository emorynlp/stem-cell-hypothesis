# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-24 17:42
from elit.components.mtl.gated.joint_gated_mtl import JointGatedMultiTaskLearning
from stem_cell_hypothesis import cdroot

cdroot()
mtl = JointGatedMultiTaskLearning()
save_dir = f'data/model/mtl/ontonotes_electra_base_en/joint_gated/pos_dep/finetune/tune'
mtl.load(save_dir, devices=-1)
print(mtl.get_gates('pos'))
print(mtl.get_sparsity_rate('pos'))