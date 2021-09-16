# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-11 21:59
text = '''
dimgray

#696969

seagreen

#2e8b57

maroon

#800000

midnightblue

#191970

olive

#808000

red

#ff0000

darkorange

#ff8c00

gold

#ffd700

mediumorchid

#ba55d3

mediumspringgreen

#00fa9a

aqua

#00ffff

blue

#0000ff

greenyellow

#adff2f

fuchsia

#ff00ff

dodgerblue

#1e90ff

salmon

#fa8072

palegoldenrod

#eee8aa

plum

#dda0dd

deeppink

#ff1493

lightskyblue

#87cefa
'''

colors = [x for x in text.splitlines() if x.startswith('#')]
print(colors)