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
# mc_iterations = 100
mc_iterations = 10
max_t = 300

# res = 10
res = 20
# interval = (0.008,0.03)
# interval = (p,0.075)
interval = (p,6*p)
# this means <res> equidistant points on <interval>


working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

# the plots are created in cache folder
a = time.time()

# Cs, unsuccessful_flag,peak_times, peak_heights,period_prevalences = vary_C(res,n,p,p_i,mc_iterations,max_t,interval,mode='tracing',force_recompute=force_recompute,path=path)

if __name__ == '__main__':

    with HiddenPrints():
        Cs, unsuccessful_flags_1,peak_times_1, peak_heights_1,period_prevalences_1, \
        Cs, unsuccessful_flags_2,peak_times_2, peak_heights_2,period_prevalences_2,\
        Cs, unsuccessful_flags_3,peak_times_3, peak_heights_3,period_prevalences_3, achieved_C, achieved_D = \
            vary_C_comp_epcurves(res, n, p, p_i, mc_iterations, max_t, interval,seed=1, force_recompute=force_recompute, path=path)



b = time.time()


# print([Cs, unsuccessful_flag,peak_times, peak_heights,period_prevalences])
# print('Time:{} seconds'.format(b-a))


