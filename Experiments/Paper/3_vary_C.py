from net import Net
import pickle
from do_experiment import *
import networkx as nx
import numpy as np
import random
import os

# 3rd experiment:
#   1) vary the clustering coefficient
#   2) pick the lowest feasible p and fix it
#   3) generate a table or plot



n = 500
p = 0.008
p_i = 0.5
mc_iterations = 10
max_t = 200

res = 5
# interval = (0.008,0.03)
interval = (0.1*p,5*p)
# this means <res> equidistant points on <interval>


working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

# the plots are created in cache folder
vary_C(res,n,p,p_i,mc_iterations,max_t,interval,mode='tracing',force_recompute=False,path=path)
