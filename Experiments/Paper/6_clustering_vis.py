import argparse
import time

from net import Net
import pickle
from do_experiment import *
import networkx as nx
import numpy as np
import random
import os
import sys
from helpers import HiddenPrints

# visualize networks with high clustering vs low clustering

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
args = parser.parse_args()
force_recompute = args.recompute

n = 500
p = 0.01
p_i = 0.5
max_t = 300

interval = (p, 6 * p)

columwidth = 251.8 / 72.27  # 251.80688[pt] / 72.27[pt/inch] FOR PAPER VISUALIZATIONS
# Direct input
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"
# plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
# Options
params = {'text.usetex': True,
          'font.size': 10,
          # 'font.family' : 'lmodern',
          }
plt.rcParams.update(params)

working_dir = os.getcwd()
path = os.path.join(working_dir, 'Pics')

# the plots are created in cache folder
a = time.time()

# same seed so before clustering is applied, both should be the same
net1 = Net(n, p, p_i, max_t, 1, clustering_target=interval[0])
net2 = Net(n, p, p_i, max_t, 1, clustering_target=interval[1])

plt.figure(figsize=(columwidth,2*columwidth),dpi=1000)

plt.subplot(211)
net1.draw(show=False)
plt.title('Graph $g$ with $C(g) = {}$'.format(interval[0]))
print(len(net1.graph.edges))

plt.subplot(212)
net2.draw(show=False)
plt.title('Graph $g$ with $C(g) = {}$'.format(interval[1]))
print(len(net2.graph.edges))



# plt.show()

plt.savefig(os.path.join(path, 'network_clustering_vis' + '.jpg'), bbox_inches='tight')


b = time.time()

print('Time:{} seconds'.format(b - a))
