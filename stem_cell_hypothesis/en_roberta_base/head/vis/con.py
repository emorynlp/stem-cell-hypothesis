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
NP	22.95	27.02 ± 0.05	27.25 ± 0.30	27.15 ± 0.97	27.03 ± 0.04	28.26 ± 0.73	31.03 ± 0.48
ADVP	35.43	26.22 ± 1.84	25.43 ± 1.39	23.12 ± 1.15	24.56 ± 3.23	21.67 ± 1.08	18.98 ± 0.68
ADJP	53.53	45.93 ± 0.68	45.28 ± 0.93	46.15 ± 0.52	45.58 ± 0.22	44.65 ± 0.30	45.06 ± 0.98
VP	52.66	49.53 ± 0.74	49.88 ± 0.44	49.70 ± 0.61	49.53 ± 0.63	48.37 ± 1.35	49.47 ± 0.73
NML	64.20	67.84 ± 1.52	67.24 ± 1.01	66.80 ± 1.49	67.33 ± 0.40	67.32 ± 0.82	68.00 ± 0.51
WHNP	80.43	53.07 ± 4.40	47.44 ± 3.41	47.50 ± 1.26	45.75 ± 0.62	45.14 ± 0.29	44.96 ± 0.44
INTJ	63.30	57.51 ± 3.09	60.01 ± 2.45	54.46 ± 3.97	56.29 ± 1.65	55.85 ± 1.50	53.34 ± 1.20
QP	78.00	86.69 ± 0.66	87.60 ± 0.82	86.83 ± 0.66	87.99 ± 0.58	87.17 ± 0.42	86.59 ± 0.73
WHADVP	49.64	50.32 ± 2.89	49.64 ± 1.91	48.96 ± 1.38	52.35 ± 3.20	50.70 ± 2.61	47.55 ± 1.40
PRT	80.06	70.74 ± 1.06	69.15 ± 3.43	65.47 ± 2.79	69.00 ± 4.04	70.04 ± 0.75	66.77 ± 0.17
PP	50.00	44.52 ± 1.81	46.35 ± 2.77	46.12 ± 0.40	44.98 ± 2.60	44.52 ± 1.81	46.12 ± 3.09
CONJP	82.19	82.19 ± 2.37	83.56 ± 1.37	84.47 ± 0.79	84.02 ± 0.79	81.28 ± 2.85	82.65 ± 2.09
X	65.85	52.85 ± 1.41	53.66 ± 2.44	60.97 ± 4.22	56.10 ± 2.44	55.29 ± 1.41	54.47 ± 3.73
WHADJP	80.56	82.41 ± 1.60	82.41 ± 1.60	82.41 ± 1.60	79.63 ± 1.61	80.56 ± 2.78	84.26 ± 1.61
META	91.67	91.67 ± 0.00	93.06 ± 2.40	91.67 ± 0.00	91.67 ± 0.00	91.67 ± 0.00	91.67 ± 0.00
UCP	94.74	84.21 ± 0.00	85.96 ± 3.04	84.21 ± 5.26	84.21 ± 0.00	82.46 ± 3.04	87.72 ± 3.04
S	94.44	94.44 ± 0.00	94.44 ± 0.00	94.44 ± 0.00	94.44 ± 0.00	94.44 ± 0.00	94.44 ± 0.00
LST	94.12	94.12 ± 0.00	94.12 ± 0.00	94.12 ± 0.00	96.08 ± 3.39	94.12 ± 0.00	94.12 ± 0.00
FRAG	66.67	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00
SBAR	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
SQ	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
WHPP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
TOP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['RoBERTa', 'POS', 'NER', 'DEP', 'SRL', 'MTL-5']
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
    pdf = 'EMNLP-2021-MTL/fig/roberta/con-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
