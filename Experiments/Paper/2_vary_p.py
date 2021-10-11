import argparse

from net import Net
import pickle
from do_experiment import *
import networkx as nx
import numpy as np
import random
import os
from helpers import HiddenPrints

# 2nd experiment:
#   1) see the effects of varying the connectivity of the network
#   2) compare between scenarios: vanilla vs quarantine vs tracing
#   3) visualize via tables/heatmaps the effect of the latter two on the prevalence

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
args = parser.parse_args()
force_recompute = args.recompute

n = 500
p_i = 0.5
mc_iterations = 100
max_t = 200

res = 50
# interval = (0.008,0.03)
interval = (0.008,0.1)
# this means <res> equidistant points on <interval>


working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

# the plots are created in cache folder
# vary_p(res, n, p_i, mc_iterations, max_t,interval=interval, mode=None, force_recompute=force_recompute, path=path)
# vary_p(res, n, p_i, mc_iterations, max_t,interval=interval, mode='quarantine', force_recompute=force_recompute, path=path)
# vary_p(res, n, p_i, mc_iterations, max_t,interval=interval, mode='tracing', force_recompute=force_recompute, path=path)
if __name__ == '__main__':
    with HiddenPrints():
        vary_p_plot_cache(res, n, p_i, mc_iterations, max_t, interval=interval, force_recompute=force_recompute, path=path)

