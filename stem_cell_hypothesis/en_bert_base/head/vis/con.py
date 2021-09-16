# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 19:50
from collections import defaultdict

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams.update({'font.size': 8})

def main():
    table = '''
	static	single	pos	ner	dep	srl	joint
NP	55.97	85.72 ± 3.76	65.53 ± 5.58	72.55 ± 14.64	67.25 ± 5.42	71.80 ± 14.06	57.58 ± 7.29
ADVP	68.61	68.61 ± 10.83	70.25 ± 5.30	77.86 ± 1.86	70.47 ± 6.46	61.15 ± 2.69	66.59 ± 5.09
ADJP	53.25	67.83 ± 3.98	69.33 ± 5.71	65.40 ± 8.79	57.69 ± 5.49	64.42 ± 1.09	63.43 ± 1.19
VP	62.83	79.29 ± 6.66	66.39 ± 4.78	70.30 ± 16.04	78.07 ± 4.17	54.85 ± 1.34	69.70 ± 5.32
NML	74.89	82.92 ± 0.97	81.51 ± 1.95	82.17 ± 0.89	81.66 ± 1.01	84.24 ± 0.68	76.02 ± 1.25
WHNP	62.26	77.87 ± 3.63	75.12 ± 1.79	73.27 ± 1.20	78.82 ± 1.71	70.63 ± 1.36	71.14 ± 1.55
INTJ	72.67	76.61 ± 1.21	84.84 ± 2.97	78.11 ± 3.74	68.79 ± 5.51	73.13 ± 4.10	74.11 ± 2.90
QP	90.45	94.55 ± 0.74	93.49 ± 0.88	93.73 ± 0.89	91.70 ± 1.01	91.65 ± 0.93	88.28 ± 0.44
WHADVP	69.43	78.99 ± 2.36	80.16 ± 6.42	75.84 ± 7.43	84.04 ± 2.50	72.63 ± 5.92	76.95 ± 2.60
PRT	79.46	80.85 ± 1.22	80.90 ± 4.60	81.20 ± 4.05	83.63 ± 3.42	73.71 ± 3.81	76.34 ± 2.78
PP	69.86	73.52 ± 1.72	79.00 ± 3.52	78.31 ± 2.09	72.83 ± 3.77	69.86 ± 12.50	71.00 ± 5.23
CONJP	91.78	90.87 ± 5.54	90.41 ± 2.37	85.84 ± 0.79	88.58 ± 1.58	88.13 ± 3.16	87.67 ± 4.11
X	60.98	59.35 ± 5.08	52.85 ± 1.41	53.66 ± 0.00	57.73 ± 2.82	55.29 ± 2.82	56.91 ± 3.73
WHADJP	91.67	95.37 ± 1.61	93.52 ± 1.60	96.29 ± 1.61	94.44 ± 2.78	94.44 ± 5.56	91.67 ± 2.78
META	95.83	98.61 ± 2.41	97.22 ± 2.41	95.83 ± 0.00	97.22 ± 2.41	97.22 ± 2.41	98.61 ± 2.41
UCP	100.00	100.00 ± 0.00	98.25 ± 3.04	100.00 ± 0.00	98.25 ± 3.04	100.00 ± 0.00	100.00 ± 0.00
S	94.44	96.29 ± 3.21	96.29 ± 3.21	94.44 ± 0.00	96.29 ± 3.21	98.15 ± 3.21	94.44 ± 0.00
LST	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
FRAG	100.00	83.33 ± 16.67	100.00 ± 0.00	94.44 ± 9.62	94.44 ± 9.62	88.89 ± 9.62	100.00 ± 0.00
SBAR	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
SQ	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
WHPP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
TOP	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00    
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['BERT', 'POS', 'NER', 'DEP', 'SRL', 'MTL-5']
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
    plt.savefig('EMNLP-2021-MTL/fig/con-acc-diff.pdf')
    plt.show()


if __name__ == '__main__':
    main()
