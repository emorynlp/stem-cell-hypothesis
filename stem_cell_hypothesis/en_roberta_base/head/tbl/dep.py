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
	static	single	pos	ner	con	srl	joint
prep	72.62	72.80 ± 1.41	69.98 ± 0.90	70.90 ± 1.02	73.11 ± 4.87	76.60 ± 9.03	78.02 ± 1.90
pobj	96.35	95.64 ± 0.69	95.01 ± 0.64	91.70 ± 0.19	94.67 ± 0.84	93.45 ± 0.70	88.59 ± 0.70
nsubj	76.36	76.92 ± 3.26	66.45 ± 3.27	73.78 ± 3.73	73.75 ± 5.32	85.61 ± 3.23	70.69 ± 4.65
det	93.70	94.07 ± 0.37	93.22 ± 0.17	93.87 ± 0.84	95.68 ± 1.25	95.72 ± 0.20	91.66 ± 0.23
root	96.25	96.28 ± 0.03	95.83 ± 0.68	96.49 ± 0.18	95.10 ± 0.94	97.77 ± 1.93	95.18 ± 1.82
nn	89.47	89.08 ± 0.82	84.77 ± 1.02	86.99 ± 0.59	92.59 ± 0.13	89.99 ± 2.44	89.98 ± 0.65
amod	93.33	92.76 ± 0.50	91.38 ± 0.74	91.63 ± 0.73	95.36 ± 0.26	92.70 ± 1.34	93.50 ± 0.67
dobj	94.55	93.76 ± 0.75	94.88 ± 0.39	93.60 ± 1.58	93.46 ± 0.50	92.95 ± 2.01	89.07 ± 1.59
advmod	69.12	70.58 ± 1.60	66.11 ± 1.89	71.85 ± 1.86	69.23 ± 2.06	68.96 ± 6.33	67.92 ± 0.93
aux	84.53	84.70 ± 0.47	84.03 ± 0.54	85.78 ± 0.39	84.22 ± 0.53	86.06 ± 0.56	84.02 ± 0.44
cc	57.86	59.17 ± 1.52	58.10 ± 2.55	58.73 ± 2.37	57.11 ± 0.88	54.40 ± 1.98	64.18 ± 3.79
conj	64.82	66.66 ± 1.65	43.20 ± 5.26	50.76 ± 2.29	65.78 ± 5.04	48.50 ± 18.81	43.52 ± 1.75
dep	42.77	41.44 ± 1.46	32.82 ± 2.05	41.88 ± 2.04	41.22 ± 2.84	42.01 ± 5.60	38.78 ± 2.47
poss	77.96	78.20 ± 0.21	79.99 ± 2.06	81.39 ± 0.67	77.72 ± 1.60	72.42 ± 6.75	79.02 ± 0.76
ccomp	71.47	71.24 ± 4.96	50.24 ± 7.03	69.80 ± 8.03	65.29 ± 11.10	66.38 ± 15.01	61.40 ± 10.14
cop	87.09	88.80 ± 1.58	87.47 ± 0.75	89.91 ± 2.45	88.74 ± 2.12	88.04 ± 3.78	85.88 ± 0.89
mark	90.88	89.78 ± 1.55	89.44 ± 1.64	91.01 ± 1.00	90.29 ± 0.76	87.65 ± 2.63	89.75 ± 0.90
xcomp	70.53	68.11 ± 2.20	65.81 ± 3.02	72.80 ± 4.66	73.20 ± 7.86	73.33 ± 9.27	71.54 ± 1.72
num	87.61	84.94 ± 2.32	83.58 ± 2.47	87.48 ± 1.18	91.50 ± 0.85	88.02 ± 2.08	90.49 ± 1.49
rcmod	52.62	51.72 ± 2.35	41.62 ± 3.40	41.96 ± 1.97	52.22 ± 7.16	59.36 ± 22.47	51.52 ± 5.04
advcl	54.93	52.26 ± 2.41	41.78 ± 1.07	53.70 ± 2.81	49.97 ± 4.71	42.06 ± 9.63	48.50 ± 0.76
neg	83.40	82.06 ± 1.41	81.66 ± 2.42	82.67 ± 0.70	84.82 ± 1.51	81.09 ± 3.26	78.43 ± 1.36
auxpass	97.51	97.37 ± 0.13	97.18 ± 0.34	97.26 ± 0.17	95.74 ± 0.32	95.66 ± 2.22	96.24 ± 0.42
nsubjpass	83.76	79.34 ± 4.75	73.26 ± 2.85	79.43 ± 6.40	73.45 ± 1.16	77.24 ± 9.83	74.59 ± 1.96
possessive	99.23	99.23 ± 0.00	99.23 ± 0.00	99.20 ± 0.06	99.26 ± 0.06	99.20 ± 0.06	99.33 ± 0.10
pcomp	90.06	87.55 ± 2.46	83.03 ± 0.30	86.95 ± 2.47	85.11 ± 1.67	84.57 ± 0.87	80.15 ± 0.60
discourse	73.80	74.04 ± 3.50	50.24 ± 14.62	62.18 ± 4.71	56.72 ± 6.79	73.14 ± 17.91	52.05 ± 3.16
partmod	59.47	60.08 ± 0.53	59.52 ± 0.32	60.49 ± 1.63	63.14 ± 0.53	64.85 ± 6.55	62.58 ± 0.44
appos	47.88	54.16 ± 5.48	41.85 ± 4.66	47.30 ± 2.18	54.65 ± 4.00	50.12 ± 11.81	43.55 ± 2.56
prt	95.99	96.44 ± 1.04	96.39 ± 0.31	96.69 ± 1.35	96.83 ± 0.22	95.30 ± 4.07	96.09 ± 0.31
number	78.43	80.75 ± 2.40	77.71 ± 0.88	82.86 ± 1.83	81.12 ± 3.15	81.41 ± 6.31	82.57 ± 2.47
quantmod	73.06	75.49 ± 2.22	72.65 ± 2.04	78.32 ± 2.48	80.18 ± 1.89	73.06 ± 0.64	75.41 ± 0.56
parataxis	43.92	46.48 ± 2.97	30.69 ± 7.25	45.24 ± 3.47	43.30 ± 2.91	47.79 ± 3.09	33.69 ± 1.80
infmod	71.20	71.38 ± 0.31	68.35 ± 2.42	70.13 ± 3.98	70.49 ± 2.68	73.24 ± 9.10	68.18 ± 3.07
tmod	87.64	84.89 ± 2.62	69.23 ± 2.89	78.02 ± 4.07	76.01 ± 3.71	80.31 ± 5.34	62.64 ± 5.00
expl	86.27	85.73 ± 0.94	86.93 ± 2.91	86.82 ± 2.78	84.75 ± 1.00	84.75 ± 0.76	84.10 ± 0.50
mwe	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00
npadvmod	87.16	86.24 ± 3.31	88.07 ± 2.30	82.42 ± 0.53	84.25 ± 2.53	83.03 ± 3.58	81.19 ± 3.21
iobj	94.02	93.30 ± 1.75	91.12 ± 1.26	89.67 ± 1.63	91.85 ± 0.00	93.66 ± 4.39	88.95 ± 2.06
predet	91.30	91.72 ± 0.36	91.93 ± 2.49	93.17 ± 1.87	92.34 ± 2.87	87.78 ± 8.01	90.47 ± 0.95
acomp	89.17	89.60 ± 0.74	88.33 ± 0.37	89.60 ± 0.37	89.81 ± 0.64	87.05 ± 4.24	89.17 ± 0.64
csubj	57.76	56.90 ± 1.50	57.19 ± 4.75	59.48 ± 2.28	53.45 ± 5.24	51.72 ± 13.27	55.17 ± 4.31
preconj	78.26	76.81 ± 2.51	65.22 ± 4.35	72.46 ± 9.81	82.61 ± 5.75	83.33 ± 6.64	75.36 ± 7.64
csubjpass	100.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	100.00 ± 0.00	77.78 ± 19.24	66.67 ± 0.00    
    '''
    head = True
    group = defaultdict(dict)
    names = ['BERT', 'POS', 'NER', 'CON', 'SRL', 'MTL-5']
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
    exit(0)

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
    plt.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    plt.legend(bbox_to_anchor=(1, 1.01), loc='upper left', labelspacing=0.2, borderpad=0.1, handletextpad=0.1)
    plt.xticks(list(range(1, 1 + len(group))), list(group.keys()))
    # plt.ylabel('Δacc')
    plt.tight_layout()
    plt.savefig('EMNLP-2021-MTL/fig/ner-acc-diff.pdf')
    plt.show()


if __name__ == '__main__':
    main()
