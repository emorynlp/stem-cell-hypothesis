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
NP	21.50	26.87 ± 0.28	30.76 ± 2.02	28.78 ± 2.34	29.98 ± 0.90	30.50 ± 2.23	30.86 ± 1.56
ADVP	14.67	26.66 ± 5.47	22.80 ± 1.36	26.03 ± 11.81	19.94 ± 1.54	22.79 ± 4.35	22.79 ± 6.43
ADJP	52.85	42.28 ± 0.87	42.21 ± 1.42	45.74 ± 2.51	43.56 ± 2.48	41.80 ± 2.08	42.18 ± 2.23
VP	49.01	49.72 ± 1.36	50.47 ± 0.46	49.85 ± 2.17	52.24 ± 1.58	52.04 ± 2.02	50.84 ± 4.08
NML	68.04	69.21 ± 0.50	67.50 ± 0.79	68.37 ± 1.11	69.92 ± 1.73	67.88 ± 3.12	66.87 ± 2.03
WHNP	56.89	58.54 ± 4.81	54.90 ± 6.65	50.85 ± 4.53	50.98 ± 4.77	52.56 ± 4.71	50.33 ± 1.41
INTJ	63.89	53.60 ± 1.78	55.44 ± 4.31	59.50 ± 5.69	52.58 ± 4.98	55.35 ± 3.12	52.66 ± 1.59
QP	79.02	89.77 ± 1.48	90.11 ± 0.96	90.01 ± 0.87	90.93 ± 0.93	90.60 ± 1.25	88.18 ± 2.25
WHADVP	49.64	52.01 ± 0.99	51.77 ± 5.42	50.95 ± 4.23	52.84 ± 1.60	51.09 ± 1.51	51.19 ± 2.24
PRT	65.48	68.25 ± 1.38	71.33 ± 3.25	74.01 ± 3.18	69.15 ± 1.19	73.21 ± 4.93	74.11 ± 7.86
PP	52.05	51.14 ± 1.43	55.25 ± 2.77	52.05 ± 0.69	50.46 ± 2.09	50.69 ± 4.28	52.28 ± 1.04
CONJP	84.93	86.76 ± 3.45	90.87 ± 3.16	86.30 ± 2.37	91.32 ± 3.45	93.15 ± 0.00	90.87 ± 3.16
X	39.02	39.02 ± 0.00	45.53 ± 11.27	42.27 ± 3.73	44.71 ± 5.08	42.27 ± 3.73	49.59 ± 5.08
WHADJP	86.11	86.11 ± 2.78	84.26 ± 3.20	83.33 ± 5.56	86.11 ± 2.78	87.04 ± 1.61	86.11 ± 2.78
META	95.83	95.83 ± 0.00	95.83 ± 0.00	95.83 ± 0.00	95.83 ± 0.00	94.44 ± 2.40	95.83 ± 0.00
UCP	84.21	84.21 ± 0.00	84.21 ± 5.26	87.72 ± 3.04	85.96 ± 3.04	85.96 ± 3.04	87.72 ± 3.04
S	94.44	94.44 ± 0.00	90.74 ± 3.20	94.44 ± 0.00	90.74 ± 3.20	94.44 ± 0.00	94.44 ± 0.00
LST	100.00	94.12 ± 0.00	94.12 ± 0.00	96.08 ± 3.39	94.12 ± 0.00	94.12 ± 0.00	94.12 ± 0.00
FRAG	66.67	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00	66.67 ± 0.00
SBAR	80.00	100.00 ± 0.00	93.33 ± 11.55	93.33 ± 11.55	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
SQ	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
WHPP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
TOP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['DeBERTa', 'POS', 'NER', 'DEP', 'SRL', 'MTL-5']
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
    pdf = 'EMNLP-2021-MTL/fig/deberta/con-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
