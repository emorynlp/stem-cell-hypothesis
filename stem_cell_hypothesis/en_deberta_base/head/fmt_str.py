# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-03-25 19:12

text = '''
ARG1    74.90
ARG0    71.90
ARG2    87.37
ARGM-TMP        66.98
ARGM-DIS        63.52
ARGM-ADV        53.63
ARGM-MOD        77.63
ARGM-LOC        75.23
ARGM-MNR        76.99
ARGM-NEG        88.88
R-ARG1  73.13
R-ARG0  78.02
C-ARG1  50.10
ARGM-PRP        79.05
ARGM-DIR        95.00
ARG3    89.30
ARG4    93.68
ARGM-CAU        71.19
ARGM-PRD        58.75
ARGM-ADJ        84.00
ARGM-EXT        87.58
ARGM-PNC        80.26
ARGM-GOL        87.67
ARGM-LVB        95.77
R-ARGM-LOC      78.46
R-ARGM-TMP      74.60
R-ARG2  75.81
C-ARG2  46.94
C-ARG0  52.94
ARGM-REC        88.24
ARGM-COM        85.19
C-ARGM-ADV      63.64
R-ARGM-MNR      72.73
ARG5    100.00
C-ARGM-TMP      57.14
R-ARGM-CAU      100.00
C-ARGM-CAU      66.67
C-ARG3  100.00
R-ARG4  100.00
C-ARGM-EXT      100.00
R-ARGM-ADV      100.00
C-ARGM-LOC      100.00
C-ARGM-MNR      100.00
C-ARGM-MOD      100.00
R-ARGM-PRP      100.00
ARGM-PRX        100.00
ARGA    100.00
R-ARGM-DIR      100.00
R-ARGM-PRD      100.00
C-ARGM-PRD      100.00
R-ARGM-EXT      100.00
C-ARGM-PRP      100.00
R-ARG3  100.00
ARGM-PRR        100.00
ARGM-DSP        100.00
'''

for line in text.split('\n'):
    line = line.strip()
    if not line:
        continue
    print('\t'.join(line.split()))
