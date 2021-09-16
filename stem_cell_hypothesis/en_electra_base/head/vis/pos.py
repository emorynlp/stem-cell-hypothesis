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
NN	56.88	37.96 ± 0.53	34.37 ± 0.55	32.01 ± 2.63	32.80 ± 2.33	33.07 ± 1.32	27.21 ± 4.88
IN	26.15	26.84 ± 3.71	24.10 ± 6.38	24.74 ± 12.25	21.33 ± 1.00	24.60 ± 1.35	19.90 ± 4.01
DT	61.31	55.67 ± 2.16	56.62 ± 0.90	54.55 ± 0.93	54.94 ± 0.79	55.86 ± 1.70	51.43 ± 1.01
NNP	49.27	35.33 ± 0.26	32.27 ± 2.02	31.89 ± 0.89	33.83 ± 1.36	32.65 ± 0.95	30.14 ± 2.22
JJ	46.28	31.98 ± 1.07	27.48 ± 1.48	26.94 ± 0.79	27.35 ± 0.13	25.67 ± 0.74	22.76 ± 2.55
.	96.83	96.42 ± 0.37	95.60 ± 1.13	94.41 ± 0.34	95.58 ± 0.18	95.30 ± 0.30	94.79 ± 1.00
NNS	71.05	56.68 ± 1.19	50.07 ± 1.59	49.16 ± 0.70	50.12 ± 1.35	50.83 ± 2.51	41.43 ± 3.56
PRP	71.46	59.27 ± 2.05	59.80 ± 3.47	55.82 ± 3.02	56.73 ± 3.27	54.13 ± 2.72	53.43 ± 1.18
RB	47.59	42.33 ± 2.14	42.64 ± 1.72	39.31 ± 2.97	37.79 ± 3.93	34.93 ± 1.71	29.15 ± 1.71
,	81.45	97.11 ± 0.60	95.43 ± 1.86	95.34 ± 1.16	93.85 ± 1.87	95.17 ± 1.64	92.85 ± 0.95
VB	73.64	73.25 ± 1.50	72.76 ± 1.75	74.39 ± 0.97	73.15 ± 1.89	71.17 ± 3.17	72.29 ± 2.54
VBD	52.84	36.42 ± 7.89	33.00 ± 2.53	33.19 ± 5.36	32.17 ± 1.26	32.48 ± 2.85	26.15 ± 3.53
CC	60.11	46.28 ± 1.63	50.07 ± 3.19	48.91 ± 4.49	45.99 ± 3.90	44.75 ± 7.39	38.16 ± 3.31
VBZ	55.89	31.56 ± 0.64	32.29 ± 0.96	32.38 ± 5.58	35.05 ± 1.80	34.09 ± 3.79	32.79 ± 13.23
VBP	55.89	52.92 ± 2.52	55.21 ± 1.15	55.52 ± 0.25	54.73 ± 0.97	54.96 ± 0.52	53.28 ± 2.55
VBN	63.66	47.73 ± 0.62	42.69 ± 0.91	41.82 ± 0.37	42.28 ± 1.63	44.50 ± 0.81	35.18 ± 0.30
CD	74.12	57.70 ± 1.97	49.45 ± 2.32	52.09 ± 2.29	53.23 ± 2.81	55.82 ± 0.97	45.10 ± 3.65
VBG	61.73	45.46 ± 3.53	41.57 ± 1.60	41.01 ± 1.43	43.00 ± 1.92	38.83 ± 2.78	31.73 ± 2.52
TO	74.09	84.27 ± 4.72	77.58 ± 2.83	76.96 ± 2.61	74.95 ± 0.12	74.85 ± 0.56	73.61 ± 1.07
MD	86.16	69.63 ± 0.63	68.27 ± 0.89	65.69 ± 0.90	64.67 ± 3.52	65.12 ± 1.61	58.64 ± 2.64
PRP$	79.84	67.10 ± 6.43	60.05 ± 3.21	61.09 ± 6.28	59.00 ± 3.76	58.26 ± 1.31	51.37 ± 2.30
UH	73.77	67.06 ± 2.49	67.12 ± 1.88	66.75 ± 3.98	66.63 ± 5.28	67.20 ± 1.74	66.35 ± 1.81
HYPH	97.03	84.76 ± 2.67	83.61 ± 4.36	78.38 ± 2.16	83.12 ± 1.87	86.77 ± 2.52	79.55 ± 7.41
POS	97.99	93.49 ± 0.00	93.49 ± 0.00	93.49 ± 0.00	93.49 ± 0.00	93.49 ± 0.00	93.49 ± 0.00
'	67.14	61.19 ± 3.80	50.75 ± 4.24	57.58 ± 2.72	58.40 ± 1.83	60.21 ± 1.10	54.15 ± 2.16
``	72.70	54.57 ± 5.54	53.71 ± 2.01	56.14 ± 3.27	48.72 ± 4.35	56.10 ± 7.53	58.53 ± 6.54
WDT	61.05	51.73 ± 4.90	50.14 ± 5.56	50.67 ± 7.68	46.48 ± 1.36	46.07 ± 3.13	48.11 ± 5.17
WP	84.66	61.51 ± 3.82	56.49 ± 0.57	57.02 ± 2.17	59.47 ± 3.20	59.06 ± 4.06	52.49 ± 2.73
WRB	63.27	59.82 ± 2.81	60.71 ± 1.74	56.42 ± 3.27	52.61 ± 0.91	56.10 ± 2.64	50.88 ± 8.43
RP	80.99	76.55 ± 0.46	75.74 ± 0.15	74.71 ± 0.70	75.17 ± 0.86	74.80 ± 0.95	73.71 ± 0.43
:	61.24	53.81 ± 1.41	48.75 ± 2.66	45.01 ± 2.54	43.76 ± 4.03	50.19 ± 1.69	42.57 ± 8.32
JJR	52.14	48.93 ± 0.74	49.08 ± 0.33	45.37 ± 1.50	47.72 ± 1.55	48.15 ± 0.33	42.52 ± 2.78
NNPS	56.64	54.18 ± 0.13	50.62 ± 0.33	52.58 ± 1.03	51.27 ± 0.13	53.01 ± 1.03	49.60 ± 2.76
EX	93.16	89.79 ± 2.63	92.07 ± 1.14	88.71 ± 2.73	90.55 ± 1.98	91.20 ± 1.17	86.97 ± 3.43
JJS	59.83	43.96 ± 0.67	43.23 ± 3.41	42.36 ± 0.44	46.00 ± 4.38	42.65 ± 2.91	41.49 ± 1.51
RBR	80.27	51.72 ± 5.96	47.39 ± 1.37	44.84 ± 5.17	43.20 ± 5.85	44.39 ± 1.18	46.49 ± 3.92
-LRB-	84.26	82.74 ± 4.43	72.93 ± 3.84	82.57 ± 5.95	86.47 ± 1.17	84.77 ± 2.69	82.91 ± 2.05
-RRB-	86.22	84.18 ± 1.35	84.52 ± 1.18	82.31 ± 0.29	84.35 ± 2.36	84.69 ± 1.53	85.38 ± 5.51
$	94.80	94.80 ± 1.53	91.52 ± 0.89	93.26 ± 1.46	94.03 ± 3.18	94.41 ± 2.19	86.90 ± 4.49
PDT	87.95	86.15 ± 4.55	86.54 ± 1.25	87.15 ± 0.92	87.55 ± 2.44	82.33 ± 3.92	85.14 ± 2.43
RBS	75.89	63.99 ± 3.72	61.90 ± 1.36	66.07 ± 4.72	61.90 ± 1.03	65.78 ± 1.86	59.52 ± 5.95
FW	62.89	65.64 ± 3.90	60.14 ± 1.58	62.20 ± 6.86	65.29 ± 2.14	64.61 ± 3.90	56.70 ± 5.16
NFP	56.67	61.11 ± 2.55	55.00 ± 3.33	57.78 ± 1.92	53.33 ± 3.34	60.56 ± 7.88	50.00 ± 4.41
WP$	100.00	91.23 ± 1.52	93.86 ± 4.02	90.35 ± 4.02	89.47 ± 2.64	89.47 ± 0.00	87.72 ± 5.48
XX	75.00	78.12 ± 0.00	78.12 ± 0.00	78.12 ± 3.13	80.21 ± 1.81	78.12 ± 0.00	78.12 ± 0.00
SYM	63.33	65.55 ± 3.85	67.78 ± 1.92	63.33 ± 3.34	66.67 ± 3.34	66.67 ± 0.00	64.44 ± 5.09
ADD	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
LS	100.00	100.00 ± 0.00	97.92 ± 3.61	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
AFX	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['ELECTRA', 'NER', 'DEP', 'CON', 'SRL', 'MTL-5']
    headers = ['ELECTRA', 'STL'] + names[1:]
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
    headers[1] = 'ELECTRA'
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
    pdf = 'EMNLP-2021-MTL/fig/electra/pos-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
