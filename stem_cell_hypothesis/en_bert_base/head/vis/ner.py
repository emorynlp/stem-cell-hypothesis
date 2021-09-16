# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 19:50
from collections import defaultdict

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams.update({'font.size': 8})


def main():
    table = '''
	static	single	pos	dep	con	srl	joint
GPE	76.43	91.40 ± 0.14	82.02 ± 4.14	74.70 ± 1.83	79.97 ± 6.19	84.97 ± 4.25	79.73 ± 6.15
PERSON	85.51	85.90 ± 3.64	85.38 ± 0.70	86.67 ± 3.27	83.65 ± 1.06	83.96 ± 0.67	83.60 ± 2.50
ORG	67.80	73.81 ± 2.16	74.95 ± 8.17	69.01 ± 2.23	68.04 ± 4.28	72.09 ± 1.06	69.99 ± 1.38
DATE	63.17	87.39 ± 5.68	62.51 ± 1.68	78.49 ± 6.87	73.53 ± 8.33	82.25 ± 7.47	83.37 ± 5.63
CARDINAL	66.63	76.33 ± 5.50	66.41 ± 10.63	64.81 ± 9.54	79.22 ± 1.74	73.05 ± 10.98	67.06 ± 1.96
NORP	82.28	93.34 ± 1.87	89.57 ± 0.66	88.47 ± 2.69	82.72 ± 6.28	89.58 ± 3.08	90.37 ± 2.21
PERCENT	99.71	99.81 ± 0.17	98.95 ± 0.60	99.71 ± 0.29	99.43 ± 0.29	99.90 ± 0.17	99.71 ± 0.00
MONEY	97.77	98.30 ± 0.49	96.60 ± 1.44	97.98 ± 1.11	96.60 ± 0.73	97.13 ± 1.93	96.07 ± 0.67
TIME	77.36	79.25 ± 5.91	78.14 ± 5.69	72.48 ± 1.91	78.77 ± 7.12	77.51 ± 4.53	77.05 ± 7.07
ORDINAL	89.74	90.94 ± 1.65	90.26 ± 5.36	92.65 ± 1.07	90.43 ± 1.29	88.03 ± 0.59	84.61 ± 6.43
LOC	69.83	74.86 ± 5.83	76.35 ± 1.40	72.81 ± 5.28	82.87 ± 3.27	74.68 ± 6.43	76.91 ± 3.37
WORK_OF_ART	76.51	78.11 ± 3.53	74.50 ± 4.56	76.31 ± 0.92	74.50 ± 2.51	79.52 ± 1.59	67.87 ± 0.92
FAC	77.78	82.96 ± 7.29	86.42 ± 1.87	81.73 ± 3.00	74.07 ± 5.34	85.43 ± 3.50	86.67 ± 4.12
QUANTITY	83.81	95.55 ± 1.98	95.56 ± 1.45	92.70 ± 0.55	87.62 ± 2.52	94.92 ± 2.40	89.84 ± 4.40
PRODUCT	81.58	85.09 ± 4.02	86.84 ± 3.48	88.60 ± 2.01	76.75 ± 4.23	84.65 ± 0.76	78.95 ± 5.73
EVENT	68.25	77.25 ± 3.67	74.60 ± 6.35	71.43 ± 2.75	66.14 ± 2.42	71.43 ± 1.59	70.37 ± 3.99
LAW	92.50	91.67 ± 3.82	87.50 ± 0.00	90.83 ± 1.44	86.67 ± 10.10	90.00 ± 2.50	87.50 ± 4.33
LANGUAGE	95.45	98.48 ± 2.63	100.00 ± 0.00	98.48 ± 2.63	98.48 ± 2.63	98.48 ± 2.63	98.48 ± 2.63    
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['BERT', 'POS', 'DEP', 'CON', 'SRL', 'MTL-5']
    headers = ['BERT', 'STL'] + names[1:]
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
    headers[1] = 'BERT'
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
    plt.savefig('EMNLP-2021-MTL/fig/ner-acc-diff.pdf')
    plt.show()


if __name__ == '__main__':
    main()
