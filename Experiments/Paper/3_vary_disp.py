import argparse
import time

from net import Net
import pickle
from do_experiment import *
import networkx as nx
import numpy as np
import random
import os
from helpers import HiddenPrints


# 3rd experiment:
#   1) vary the clustering coefficient
#   2) pick the lowest feasible p and fix it
#   3) generate a table or plot

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
args = parser.parse_args()
force_recompute = args.recompute

n = 500
p = 0.01
p_i = 0.5
mc_iterations = 100
max_t = 200

res = 10
# interval = (0.008,0.03)
# interval = (0.5*(1-p),20*(1-p))
interval = (0.5, 35)
# this means <res> equidistant points on <interval>


working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

# the plots are created in cache folder
a = time.time()
if __name__ == '__main__':
    with HiddenPrints():
        Disps, unsuccessful_flag,peak_times, peak_heights,period_prevalences = vary_disp(res, n, p, p_i, mc_iterations, max_t,
                                                                                         interval, mode='tracing',
                                                                                         force_recompute=force_recompute,
                                                                                         path=path)

b = time.time()


# print([Cs, unsuccessful_flag,peak_times, peak_heights,period_prevalences])
print('Time:{} seconds'.format(b-a))


