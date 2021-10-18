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
# p_i = (0.25,0.5,0.75) # AFTER CHANGES TO MODEL, THIS HAD TO BE LOWER
p_i = (0.1,0.2,0.3)

mc_iterations = 2000
max_t = 400

res = 30
interval = (p,6*p)
# this means <res> equidistant points on <interval>


working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

# the plots are created in cache folder
a = time.time()

if __name__ == '__main__':
    with HiddenPrints():
        # Cs, unsuccessful_flag,peak_times, peak_heights,period_prevalences = vary_C(res,n,p,p_i,mc_iterations,max_t,interval,mode='tracing',force_recompute=force_recompute,path=path)

        Cs, peak_times_1, peak_heights_1,period_prevalences_1, \
        peak_times_2, peak_heights_2,period_prevalences_2,\
        peak_times_3, peak_heights_3,period_prevalences_3, achieved_C, achieved_D = \
            vary_C_pi_comp_corrected(res, n, p, p_i, mc_iterations, max_t, interval,seed=0, force_recompute=force_recompute, path=path)

b = time.time()


# print([Cs, unsuccessful_flag,peak_times, peak_heights,period_prevalences])
# print('Time:{} seconds'.format(b-a))


