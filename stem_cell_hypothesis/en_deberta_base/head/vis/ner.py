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
GPE	37.86	42.25 ± 15.30	35.43 ± 1.94	40.00 ± 7.10	39.82 ± 3.71	42.13 ± 6.81	38.78 ± 2.20
PERSON	69.62	67.07 ± 2.49	67.89 ± 1.02	66.00 ± 0.89	67.19 ± 2.93	67.12 ± 3.24	69.26 ± 2.10
ORG	38.27	43.27 ± 3.32	43.21 ± 2.29	42.23 ± 2.22	42.77 ± 1.41	41.87 ± 2.32	45.94 ± 6.76
DATE	50.19	53.64 ± 6.88	47.38 ± 1.36	50.67 ± 7.50	53.68 ± 1.27	47.05 ± 1.38	47.79 ± 2.51
CARDINAL	40.53	52.12 ± 1.64	68.52 ± 2.67	51.23 ± 4.37	47.70 ± 8.63	53.65 ± 12.87	56.54 ± 14.70
NORP	58.62	57.11 ± 1.10	69.08 ± 8.79	57.79 ± 2.65	59.77 ± 1.78	58.26 ± 3.10	59.22 ± 6.96
PERCENT	91.12	96.37 ± 1.19	96.27 ± 0.99	94.94 ± 1.36	96.95 ± 1.15	95.03 ± 1.44	93.51 ± 0.72
MONEY	84.39	91.93 ± 2.12	89.17 ± 1.77	90.55 ± 0.18	88.96 ± 1.21	93.53 ± 1.33	90.24 ± 1.60
TIME	63.68	65.57 ± 0.00	68.24 ± 0.72	67.30 ± 0.98	68.55 ± 4.82	73.58 ± 10.62	68.08 ± 1.79
ORDINAL	74.36	66.33 ± 4.36	70.60 ± 4.30	71.28 ± 9.11	64.45 ± 5.44	64.10 ± 10.37	66.50 ± 9.79
LOC	44.69	48.42 ± 4.51	53.82 ± 5.19	51.40 ± 3.91	59.78 ± 18.60	49.91 ± 4.75	51.02 ± 5.28
WORK_OF_ART	56.63	54.62 ± 2.85	55.42 ± 2.17	56.22 ± 1.52	58.63 ± 3.53	58.63 ± 4.01	56.83 ± 3.04
FAC	52.59	53.58 ± 2.81	55.55 ± 4.63	53.58 ± 1.13	55.80 ± 1.71	58.03 ± 7.00	53.82 ± 1.13
QUANTITY	80.95	72.38 ± 0.95	70.48 ± 1.91	72.70 ± 3.35	75.55 ± 7.40	77.14 ± 5.95	75.56 ± 5.25
PRODUCT	48.68	52.63 ± 3.48	58.77 ± 4.02	60.09 ± 2.74	56.14 ± 4.23	55.70 ± 1.52	62.72 ± 10.05
EVENT	47.62	52.91 ± 5.10	50.79 ± 2.75	50.27 ± 1.83	51.85 ± 0.92	51.85 ± 2.42	53.97 ± 1.59
LAW	70.00	70.00 ± 4.33	70.83 ± 3.82	70.00 ± 2.50	75.00 ± 2.50	70.00 ± 2.50	74.17 ± 1.44
LANGUAGE	77.27	78.79 ± 2.63	78.79 ± 2.63	77.27 ± 4.55	77.27 ± 4.55	75.76 ± 6.94	81.82 ± 4.55	
    '''
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    head = True
    group = defaultdict(dict)
    names = ['DeBERTa', 'POS', 'DEP', 'CON', 'SRL', 'MTL-5']
    headers = ['DeBERTa', 'STL'] + names[1:]
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
    headers[1] = 'DeBERTa'
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
    pdf = 'EMNLP-2021-MTL/fig/deberta/ner-acc-diff.pdf'
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    plt.savefig(pdf)
    plt.show()


if __name__ == '__main__':
    main()
