# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 19:50
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import math

prop_cycle = plt.rcParams['axes.prop_cycle']
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams.update({'font.size': 8})


def main():
    table = '''
	static	single	ner	dep	con	srl	all
NN	31.91	32.95 ± 0.79	28.94 ± 1.26	28.44 ± 2.16	27.05 ± 1.64	28.66 ± 2.77	24.02 ± 0.86
IN	49.76	48.17 ± 0.30	30.37 ± 3.44	30.85 ± 1.75	33.92 ± 2.79	31.54 ± 4.24	21.25 ± 2.76
DT	22.60	24.53 ± 1.76	29.18 ± 3.74	32.65 ± 6.41	25.27 ± 1.75	29.95 ± 6.08	33.35 ± 2.71
NNP	58.11	57.73 ± 0.32	45.58 ± 2.16	50.83 ± 2.60	51.13 ± 2.03	50.38 ± 1.44	41.29 ± 0.60
JJ	42.78	39.15 ± 1.32	30.21 ± 1.46	32.05 ± 2.03	29.97 ± 2.81	30.33 ± 1.36	23.92 ± 2.07
.	94.15	93.10 ± 0.03	93.84 ± 0.41	94.23 ± 0.79	93.72 ± 0.84	94.73 ± 1.18	94.71 ± 1.01
NNS	49.56	47.27 ± 0.29	52.55 ± 0.47	52.79 ± 0.53	51.97 ± 0.70	51.62 ± 0.22	50.17 ± 0.24
PRP	41.25	50.43 ± 4.67	45.58 ± 3.99	42.07 ± 1.30	42.69 ± 3.03	44.90 ± 0.69	43.28 ± 3.10
RB	36.10	35.74 ± 0.28	20.31 ± 2.53	18.23 ± 3.13	20.14 ± 1.14	18.09 ± 1.37	15.38 ± 0.52
,	82.31	70.52 ± 5.24	68.53 ± 2.62	68.64 ± 0.89	68.39 ± 3.89	75.00 ± 5.12	64.57 ± 4.30
VB	68.31	68.51 ± 0.15	69.81 ± 0.74	71.84 ± 0.35	72.14 ± 0.48	72.59 ± 0.23	72.82 ± 0.70
VBD	49.05	45.05 ± 1.14	38.57 ± 6.36	41.09 ± 1.05	42.80 ± 2.19	37.91 ± 2.21	39.24 ± 0.89
CC	70.27	69.23 ± 0.94	50.28 ± 2.37	45.80 ± 3.48	42.59 ± 2.37	51.25 ± 5.52	31.29 ± 4.64
VBZ	46.20	47.37 ± 0.47	46.40 ± 2.37	41.82 ± 1.75	44.38 ± 3.03	45.82 ± 0.10	37.32 ± 1.29
VBP	54.53	54.87 ± 0.17	55.44 ± 0.41	55.21 ± 0.05	55.50 ± 0.39	54.96 ± 0.36	55.40 ± 0.31
VBN	40.27	39.34 ± 0.40	43.31 ± 2.36	42.84 ± 1.69	42.42 ± 1.91	39.95 ± 0.27	41.46 ± 1.35
CD	62.82	62.72 ± 0.43	54.55 ± 2.00	55.15 ± 0.31	54.79 ± 1.36	57.56 ± 0.42	46.97 ± 0.99
VBG	53.80	53.68 ± 1.02	48.95 ± 0.13	49.06 ± 0.14	48.95 ± 0.13	49.21 ± 0.38	48.79 ± 0.14
TO	73.89	72.73 ± 1.09	75.25 ± 1.48	75.78 ± 2.95	69.30 ± 2.12	75.11 ± 1.14	75.54 ± 3.09
MD	77.01	79.09 ± 2.82	62.97 ± 3.32	63.25 ± 0.58	63.80 ± 0.77	61.27 ± 5.04	59.92 ± 3.02
PRP$	47.29	54.44 ± 1.01	49.53 ± 5.62	49.89 ± 0.76	48.54 ± 4.42	51.27 ± 4.47	47.06 ± 5.70
UH	62.96	62.92 ± 0.50	64.02 ± 1.82	60.95 ± 0.26	60.76 ± 3.90	63.79 ± 0.88	59.66 ± 3.08
HYPH	64.59	66.57 ± 1.51	56.26 ± 3.07	61.06 ± 4.02	56.97 ± 1.62	64.99 ± 2.49	51.70 ± 3.03
POS	87.16	86.14 ± 1.94	68.46 ± 5.03	67.66 ± 7.60	60.82 ± 1.07	66.48 ± 7.54	60.92 ± 0.82
'	56.12	61.36 ± 1.10	60.41 ± 1.12	60.48 ± 1.06	60.34 ± 2.64	62.14 ± 3.63	61.80 ± 3.35
``	64.99	72.77 ± 1.50	63.88 ± 1.00	67.06 ± 3.69	64.42 ± 2.02	66.92 ± 1.39	65.02 ± 5.25
WDT	71.31	61.70 ± 1.19	44.73 ± 9.04	43.10 ± 9.27	44.08 ± 4.71	46.64 ± 4.15	44.08 ± 4.76
WP	55.24	56.69 ± 1.84	51.91 ± 6.15	53.29 ± 2.96	53.86 ± 2.94	57.31 ± 3.70	53.90 ± 3.11
WRB	52.93	53.35 ± 1.15	50.93 ± 2.91	48.28 ± 3.97	50.74 ± 4.32	49.16 ± 2.81	44.18 ± 3.50
RP	80.71	81.56 ± 0.65	70.21 ± 1.92	71.01 ± 1.40	71.82 ± 0.36	70.59 ± 0.16	69.55 ± 2.49
:	84.08	78.53 ± 1.60	63.42 ± 2.00	61.11 ± 4.70	52.43 ± 1.69	58.68 ± 3.85	52.62 ± 2.97
JJR	36.32	40.24 ± 0.75	41.10 ± 1.01	37.54 ± 0.75	39.96 ± 1.07	38.32 ± 1.94	41.38 ± 2.85
NNPS	55.99	58.10 ± 0.88	50.03 ± 0.88	53.52 ± 2.18	52.65 ± 1.77	51.49 ± 1.45	52.22 ± 2.01
EX	66.45	64.93 ± 1.00	82.08 ± 1.49	80.45 ± 2.98	86.76 ± 5.61	83.82 ± 4.93	88.49 ± 1.67
JJS	52.84	51.53 ± 1.31	38.28 ± 1.40	35.95 ± 1.82	37.41 ± 1.53	40.03 ± 1.40	38.86 ± 2.27
RBR	62.78	63.23 ± 1.18	50.67 ± 1.35	52.62 ± 2.70	47.08 ± 5.17	46.78 ± 4.15	41.86 ± 4.07
-LRB-	91.37	93.23 ± 0.59	87.81 ± 1.34	88.49 ± 3.30	82.23 ± 5.65	85.11 ± 3.06	89.68 ± 5.43
-RRB-	95.92	97.79 ± 0.29	95.92 ± 2.04	97.45 ± 0.00	96.43 ± 2.65	95.58 ± 2.12	94.56 ± 2.36
$	93.64	95.57 ± 0.66	89.98 ± 3.18	88.44 ± 1.53	89.21 ± 2.97	90.75 ± 1.16	89.60 ± 3.47
PDT	66.27	70.48 ± 1.60	74.90 ± 1.94	75.70 ± 3.63	74.50 ± 1.84	74.50 ± 4.60	78.92 ± 4.34
RBS	77.68	81.25 ± 0.89	67.86 ± 0.90	68.45 ± 0.51	66.67 ± 1.03	67.26 ± 1.36	64.88 ± 2.06
FW	72.16	72.85 ± 0.60	70.45 ± 2.38	67.70 ± 7.32	72.51 ± 1.20	71.82 ± 2.60	69.07 ± 1.03
NFP	75.00	74.44 ± 1.93	63.89 ± 2.55	65.00 ± 3.33	60.56 ± 0.96	62.22 ± 5.36	61.11 ± 6.31
WP$	94.74	93.86 ± 1.52	89.48 ± 4.56	86.84 ± 4.56	85.09 ± 4.02	90.35 ± 1.52	81.58 ± 4.56
XX	75.00	81.25 ± 0.00	78.12 ± 0.00	78.12 ± 0.00	81.25 ± 0.00	78.12 ± 0.00	81.25 ± 0.00
SYM	66.67	70.00 ± 0.00	66.67 ± 3.34	68.89 ± 3.85	63.33 ± 3.34	66.67 ± 3.34	63.33 ± 0.00
ADD	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
LS	100.00	93.75 ± 0.00	95.83 ± 3.61	91.67 ± 3.61	97.92 ± 3.61	95.83 ± 3.61	95.83 ± 3.61
AFX	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00 
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['RoBERTa', 'NER', 'DEP', 'CON', 'SRL', 'MTL-5']
    headers = ['RoBERTa', 'STL'] + names[1:]
    ss = defaultdict(dict)
    raw = defaultdict(dict)
    nmax = 20
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
        for n, s in zip(headers, scores):
            raw[n][label] = s
        scores = [float(x.split()[0]) for x in scores]
        for n, s in zip(headers, scores):
            ss[n][label] = s
        scores[0], scores[1] = scores[1], scores[0]
        for n, s in zip(names, scores[1:]):
            group[n][label] = s - scores[0]
        # nmax -= 1
        # if not nmax:
        #     break
    texts = []
    xys = defaultdict(lambda: ([], []))
    ng = set()
    for i, (n, scores) in enumerate(group.items()):
        for j, (label, diff) in enumerate(scores.items()):
            # plt.scatter(i + 1, diff)
            xys[label][0].append(i + 1)
            xys[label][1].append(diff)
            if diff > 0 and len(ng) <= 20:
                ng.add(label)
            # texts.append(plt.text(i + 1, diff, label))
    # colors = prop_cycle.by_key()['color']
    # colors.extend(['r', 'g', 'b', 'c', 'm', 'y'])
    # colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    colors = ['#696969', '#2e8b57', '#800000', '#191970', '#808000', '#ff0000', '#ff8c00', '#ffd700', '#ba55d3',
              '#00fa9a', '#00ffff', '#0000ff', '#adff2f', '#ff00ff', '#1e90ff', '#fa8072', '#eee8aa', '#dda0dd',
              '#ff1493', '#87cefa'] + ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
                                       '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
                                       '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#000000']
    print('\t'.join([''] + headers))
    headers[0] = 'STL'
    headers[1] = 'RoBERTa'
    for label in xys.keys():
        cs = ['\\texttt{' + label.replace('$', '\\$') + '}']
        win = max([(g, d[label]) for g,d in ss.items()], key=lambda x:x[-1])[1]
        for g in headers:
            data = ss[g]
            cell = f'{data[label]:.2f}'
            if win == data[label]:
                cell = '\\textbf{' + cell + '}'
            # cell = raw[g][label].replace(' ± ', '$\pm$')
            diff = group[g].get(label, None)
            if diff is not None:
                if diff > 0:
                    cell += '\\posdiff{' + f'{diff:.2f}' + '}'
                else:
                    cell += '\\negdiff{' + f'{diff:.2f}' + '}'
            cs.append(cell)
        print(' & '.join(cs) + ' \\\\ ')
    # exit(0)

    nmax = 20
    for label in xys.keys():
        if len(ng) >= nmax:
            break
        ng.add(label)
    for i, (label, xy) in enumerate(xys.items()):
        plt.scatter(*xy, label=label if label in ng else '_nolegend_', color=colors[i % len(colors)], marker='_',
                    s=300 / (i + 1))
    # adjust_text(texts)
    plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', labelspacing=0.2, borderpad=0.1, handletextpad=0.1)
    plt.xticks(list(range(1, 1 + len(group))), list(group.keys()))
    # plt.ylabel('Δacc')
    plt.tight_layout()
    pdf = 'EMNLP-2021-MTL/fig/roberta/pos-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
