# code is in net.py and montecarlo.py

from net import Net
import pickle
from do_experiment import simple_experiment
import networkx as nx
import numpy as np
import random


# testing different network params

# exp_8_path = r"C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\corner_cases"
#
# mc_iterations = 50
# max_t = 200
#
# n = 1000
# p = 0
# p_i = 0.5
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,p_i,mc_iterations, max_t, force_recompute=False, path=exp_8_path)


res = 20
n = 200
p = 0.1
p_i = 0.5
mc_iterations = 50
max_t = 200

path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments'

# from do_experiment import vary_p, vary_p_i
# path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\vary_params'
# vary_p(res=res,n=n,p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, path = path)
# vary_p(res=res,n=n,p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='quarantine', path = path)
# vary_p(res=res,n=n,p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='tracing', path = path)
#
# vary_p_i(res=res,n=n,p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, path = path)
# vary_p_i(res=res,n=n,p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='quarantine', path = path)
# vary_p_i(res=res,n=n,p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='tracing', path = path)

net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,p_i,mc_iterations, max_t, force_recompute=False, path=path)

# net.plot_timeseries(counts)

# for i in range(198):
    # net.graph.add_edge(198,i)
    # net.graph.add_edge(199,i)
    # net.graph.add_edge(197,i)
    # net.graph.add_edge(196,i)
    # net.graph.remove_edge(*random.choice(list(net.graph.edges)))
print(nx.average_clustering(net.graph))