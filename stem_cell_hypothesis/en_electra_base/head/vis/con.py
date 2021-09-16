# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 19:50
import os
from collections import defaultdict

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams.update({'font.size': 8})

def main():
    table = '''
	static	single	pos	ner	dep	srl	joint
NP	37.22	35.24 ± 0.60	34.03 ± 0.12	35.06 ± 0.72	35.12 ± 0.34	34.80 ± 0.31	33.39 ± 0.78
ADVP	63.07	54.25 ± 1.42	47.97 ± 1.79	48.75 ± 1.63	47.49 ± 2.32	48.92 ± 1.55	39.09 ± 2.31
ADJP	60.17	46.69 ± 0.82	47.59 ± 0.75	46.94 ± 0.27	46.76 ± 0.69	47.65 ± 0.86	47.67 ± 0.64
VP	69.02	52.41 ± 3.35	49.00 ± 1.21	48.31 ± 1.11	48.83 ± 0.59	48.25 ± 0.77	49.62 ± 0.66
NML	71.33	71.25 ± 0.45	71.99 ± 1.08	71.36 ± 0.74	72.35 ± 0.43	72.22 ± 0.90	72.03 ± 0.60
WHNP	66.04	60.98 ± 0.95	52.78 ± 1.32	52.82 ± 0.66	54.82 ± 0.75	53.07 ± 0.59	52.74 ± 0.80
INTJ	68.55	59.92 ± 3.15	61.25 ± 3.61	56.33 ± 1.25	58.74 ± 3.61	60.05 ± 3.20	61.82 ± 1.79
QP	90.88	91.46 ± 0.52	90.69 ± 0.33	91.22 ± 0.71	91.36 ± 1.16	90.79 ± 0.58	90.25 ± 0.55
WHADVP	59.97	53.66 ± 3.84	53.27 ± 0.25	51.68 ± 1.77	52.01 ± 0.95	52.89 ± 3.88	49.88 ± 0.97
PRT	82.59	74.60 ± 2.14	70.78 ± 0.91	72.52 ± 0.46	71.28 ± 1.12	71.53 ± 1.71	71.58 ± 1.42
PP	63.70	50.00 ± 1.81	48.63 ± 2.05	52.05 ± 1.37	52.74 ± 1.81	52.05 ± 4.28	48.40 ± 1.05
CONJP	83.56	76.71 ± 1.37	75.80 ± 0.79	78.08 ± 2.74	77.17 ± 2.85	76.71 ± 1.37	77.62 ± 2.09
X	48.78	53.66 ± 7.32	51.22 ± 2.44	49.59 ± 3.73	52.85 ± 1.41	51.22 ± 2.44	52.03 ± 3.73
WHADJP	91.67	91.67 ± 0.00	91.67 ± 0.00	91.67 ± 0.00	91.67 ± 0.00	91.67 ± 0.00	88.89 ± 0.00
META	95.83	93.06 ± 2.40	93.06 ± 2.40	91.67 ± 0.00	91.67 ± 0.00	93.06 ± 2.40	94.44 ± 2.40
UCP	78.95	87.72 ± 8.04	91.23 ± 3.04	84.21 ± 5.26	91.23 ± 3.04	85.96 ± 3.04	91.23 ± 8.04
S	94.44	96.29 ± 3.21	100.00 ± 0.00	100.00 ± 0.00	98.15 ± 3.21	96.29 ± 3.21	96.29 ± 3.21
LST	94.12	94.12 ± 0.00	94.12 ± 0.00	94.12 ± 0.00	96.08 ± 3.39	94.12 ± 0.00	94.12 ± 0.00
FRAG	66.67	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	61.11 ± 9.62	66.67 ± 0.00
SBAR	80.00	100.00 ± 0.00	100.00 ± 0.00	93.33 ± 11.55	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
SQ	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
WHPP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
TOP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['ELECTRA', 'POS', 'NER', 'DEP', 'SRL', 'MTL-5']
    nmax = 20
    c = 0
    for line in table.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = line.split('\t')
        if not cells:
            continue
        if head:
            head = False
            continue

        label, scores = cells[0], cells[1:]
        scores = [float(x.split()[0]) for x in scores]
        scores[0], scores[1] = scores[1], scores[0]
        for n, s in zip(names, scores[1:]):
            group[n][label] = s - scores[0]
        # c += 1
        # if c == nmax:
        #     break
    texts = []
    xys = defaultdict(lambda: ([], []))
    for i, (n, scores) in enumerate(group.items()):
        for j, (label, diff) in enumerate(scores.items()):
            # plt.scatter(i + 1, diff)
            xys[label][0].append(i + 1)
            xys[label][1].append(diff)
            # texts.append(plt.text(i + 1, diff, label))
    # colors = prop_cycle.by_key()['color']
    # colors.extend(['r', 'g', 'b', 'c', 'm', 'y'])
    # colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    colors = ['#696969', '#2e8b57', '#800000', '#191970', '#808000', '#ff0000', '#ff8c00', '#ffd700', '#ba55d3',
              '#00fa9a', '#00ffff', '#0000ff', '#adff2f', '#ff00ff', '#1e90ff', '#fa8072', '#eee8aa', '#dda0dd',
              '#ff1493', '#87cefa']
    for i, (label, xy) in enumerate(xys.items()):
        plt.scatter(*xy, label=label if i < nmax else '_nolegend_', color=colors[i % len(colors)], marker='_', s=300/(i+1))
    # adjust_text(texts)
    # plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', labelspacing=0.2, borderpad=0.1, handletextpad=0.1)
    plt.xticks(list(range(1, 1 + len(group))), list(group.keys()))
    # plt.ylabel('Δacc')
    plt.tight_layout()
    pdf = 'EMNLP-2021-MTL/fig/electra/con-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
