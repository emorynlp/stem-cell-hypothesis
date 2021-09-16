# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 19:50
import os
from collections import defaultdict

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams.update({'font.size': 8})


def main():
    table = '''
	static	single	pos	dep	con	srl	joint
GPE	35.62	35.83 ± 2.54	36.19 ± 1.79	37.89 ± 1.23	37.17 ± 1.72	35.62 ± 0.64	37.65 ± 1.38
PERSON	69.11	67.47 ± 0.95	68.48 ± 0.45	68.51 ± 1.16	68.83 ± 0.83	66.94 ± 1.17	69.49 ± 1.48
ORG	38.22	41.76 ± 0.61	41.58 ± 0.75	41.08 ± 0.57	41.74 ± 0.73	41.23 ± 1.49	41.39 ± 1.52
DATE	53.93	55.14 ± 1.82	48.75 ± 4.50	54.97 ± 1.61	52.16 ± 3.25	48.71 ± 1.48	47.50 ± 0.45
CARDINAL	56.15	52.26 ± 2.12	45.10 ± 2.73	46.02 ± 2.92	46.06 ± 3.10	45.31 ± 2.50	39.04 ± 0.98
NORP	58.26	54.30 ± 0.95	56.16 ± 1.91	55.37 ± 0.45	55.77 ± 1.20	53.27 ± 1.02	54.89 ± 1.67
PERCENT	92.84	94.65 ± 1.42	95.99 ± 2.28	95.80 ± 1.44	97.42 ± 1.59	97.32 ± 0.88	95.32 ± 2.33
MONEY	80.57	81.00 ± 0.49	82.48 ± 1.46	81.11 ± 1.57	82.80 ± 0.55	80.68 ± 0.49	83.12 ± 1.99
TIME	67.92	65.10 ± 2.16	62.74 ± 2.16	61.16 ± 1.18	60.85 ± 1.24	59.75 ± 0.72	57.07 ± 1.70
ORDINAL	74.36	69.06 ± 2.53	74.19 ± 3.13	74.02 ± 3.01	73.67 ± 5.34	67.69 ± 2.35	70.60 ± 4.65
LOC	45.25	46.93 ± 0.56	46.37 ± 0.56	47.11 ± 2.33	48.04 ± 2.90	48.97 ± 0.86	49.90 ± 2.33
WORK_OF_ART	51.81	51.61 ± 0.92	55.02 ± 1.25	55.22 ± 2.85	55.62 ± 1.93	55.22 ± 2.44	53.81 ± 1.52
FAC	48.89	51.85 ± 1.48	50.62 ± 1.13	52.35 ± 3.50	50.86 ± 3.34	50.37 ± 1.48	51.11 ± 1.28
QUANTITY	63.81	69.21 ± 1.10	69.84 ± 1.46	72.38 ± 1.91	69.84 ± 2.40	69.84 ± 1.98	71.43 ± 3.43
PRODUCT	56.58	54.39 ± 3.31	60.09 ± 0.76	57.89 ± 0.00	59.21 ± 1.32	58.33 ± 1.52	61.40 ± 0.76
EVENT	55.56	53.44 ± 5.10	53.97 ± 3.18	53.97 ± 2.75	55.56 ± 4.20	50.79 ± 1.59	55.56 ± 0.00
LAW	72.50	73.33 ± 1.44	74.17 ± 1.44	72.50 ± 4.33	73.33 ± 1.44	72.50 ± 4.33	72.50 ± 2.50
LANGUAGE	77.27	80.30 ± 2.63	84.85 ± 2.62	86.36 ± 4.55	84.85 ± 2.62	86.36 ± 4.55	87.88 ± 2.63	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['RoBERTa', 'POS', 'DEP', 'CON', 'SRL', 'MTL-5']
    headers = ['RoBERTa', 'STL'] + names[1:]
    ss = defaultdict(dict)
    raw = defaultdict(dict)
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
    texts = []
    xys = defaultdict(lambda: ([], []))
    for i, (n, scores) in enumerate(group.items()):
        for j, (label, diff) in enumerate(scores.items()):
            # plt.scatter(i + 1, diff)
            xys[label][0].append(i + 1)
            xys[label][1].append(diff)
            # texts.append(plt.text(i + 1, diff, label))
    nmax = 20
    colors = ['#696969', '#2e8b57', '#800000', '#191970', '#808000', '#ff0000', '#ff8c00', '#ffd700', '#ba55d3',
              '#00fa9a', '#00ffff', '#0000ff', '#adff2f', '#ff00ff', '#1e90ff', '#fa8072', '#eee8aa', '#dda0dd',
              '#ff1493', '#87cefa']

    print('\t'.join([''] + headers))
    headers[0] = 'STL'
    headers[1] = 'RoBERTa'
    for label in xys.keys():
        cs = ['\\texttt{' + label.replace('$', '\\$').replace('_', '\\_') + '}']
        win = max([(g, d[label]) for g, d in ss.items()], key=lambda x: x[-1])[1]
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

    for i, (label, xy) in enumerate(xys.items()):
        label = {
            'WORK_OF_ART': 'WOA',
            'PERSON': 'PER',
            'CARDINAL': 'CARD',
            'PERCENT': '%',
            'MONEY': '$',
            'ORDINAL': 'ORD',
            'QUANTITY': 'QUA',
            'PRODUCT': 'PRDT',
            'EVENT': 'EVNT',
            'LANGUAGE': 'LANG',
        }.get(label, label)
        plt.scatter(*xy, label=label if i < nmax else '_nolegend_', color=colors[i % len(colors)], marker='_',
                    s=300 / (i + 1))
    # adjust_text(texts)
    plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', labelspacing=0.2, borderpad=0.1, handletextpad=0.1)
    plt.xticks(list(range(1, 1 + len(group))), list(group.keys()))
    # plt.ylabel('Δacc')
    plt.tight_layout()
    pdf = 'EMNLP-2021-MTL/fig/roberta/ner-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
