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
GPE	59.51	52.68 ± 3.93	48.44 ± 2.85	48.58 ± 2.84	51.29 ± 8.66	50.77 ± 5.83	50.15 ± 1.40
PERSON	74.65	72.28 ± 0.94	72.29 ± 0.87	73.75 ± 1.92	72.60 ± 1.08	73.71 ± 0.53	71.78 ± 1.07
ORG	50.36	49.94 ± 1.41	45.76 ± 2.76	45.44 ± 3.58	45.22 ± 3.33	46.26 ± 2.03	43.12 ± 1.73
DATE	64.23	59.99 ± 3.21	52.37 ± 1.34	53.02 ± 0.81	53.72 ± 1.42	56.43 ± 2.83	46.86 ± 1.03
CARDINAL	69.73	65.85 ± 5.65	54.51 ± 6.43	51.41 ± 5.63	54.90 ± 0.59	54.37 ± 4.22	46.81 ± 1.34
NORP	67.54	62.62 ± 0.86	56.99 ± 1.31	57.39 ± 1.01	56.92 ± 0.30	58.30 ± 0.45	56.87 ± 1.61
PERCENT	100.00	99.24 ± 1.32	97.90 ± 0.92	98.19 ± 1.65	97.80 ± 2.66	98.37 ± 0.44	96.75 ± 3.64
MONEY	90.76	88.22 ± 0.84	86.41 ± 1.02	86.84 ± 0.66	87.79 ± 0.37	87.48 ± 1.12	88.32 ± 0.48
TIME	60.85	63.05 ± 5.69	61.48 ± 1.19	59.43 ± 0.82	57.55 ± 2.06	63.99 ± 3.14	57.86 ± 0.98
ORDINAL	85.64	82.05 ± 1.36	74.36 ± 4.38	78.98 ± 2.35	74.70 ± 2.07	80.17 ± 1.07	68.03 ± 2.43
LOC	52.51	57.36 ± 2.26	54.19 ± 2.95	54.94 ± 1.17	54.38 ± 0.86	56.61 ± 1.79	53.82 ± 0.86
WORK_OF_ART	66.27	58.84 ± 1.52	57.63 ± 0.92	59.04 ± 5.93	59.24 ± 2.97	59.64 ± 2.63	57.23 ± 0.60
FAC	62.96	65.93 ± 1.49	66.17 ± 3.34	66.42 ± 1.54	63.95 ± 2.81	66.42 ± 1.13	63.95 ± 1.14
QUANTITY	76.19	77.46 ± 0.55	75.87 ± 0.55	74.60 ± 1.98	75.56 ± 4.29	79.68 ± 1.10	73.97 ± 2.20
PRODUCT	59.21	64.47 ± 1.32	59.65 ± 1.52	59.65 ± 2.01	67.54 ± 4.98	60.96 ± 2.74	60.53 ± 1.32
EVENT	42.86	46.03 ± 11.11	41.80 ± 3.99	50.79 ± 9.91	51.85 ± 11.15	46.56 ± 9.70	42.33 ± 3.99
LAW	72.50	70.83 ± 1.44	74.17 ± 2.89	77.50 ± 0.00	78.33 ± 5.77	75.83 ± 5.20	75.00 ± 2.50
LANGUAGE	81.82	83.33 ± 2.62	78.79 ± 5.25	80.30 ± 2.63	80.30 ± 6.94	77.27 ± 4.55	78.79 ± 5.25	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['ELECTRA', 'POS', 'DEP', 'CON', 'SRL', 'MTL-5']
    headers = ['ELECTRA', 'STL'] + names[1:]
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
    headers[1] = 'ELECTRA'
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
    pdf = 'EMNLP-2021-MTL/fig/electra/ner-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
