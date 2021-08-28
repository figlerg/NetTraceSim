# code is in net.py and montecarlo.py
import matplotlib.pyplot as plt
import pandas as pd

from net import Net
import pickle
from do_experiment import simple_experiment
import networkx as nx
import numpy as np
import os


res = 20
n = 10
p = 0.5
p_i = 0.5
mc_iterations = 100
max_t = 200

# working_dir = os.getcwd()
# path = working_dir.join(['Experiments',])

seed = 0
net = Net(n=n, p=p, p_i=p_i, max_t=max_t, seed=seed)
net.sim(seed)

working_dir = os.getcwd()
path = os.path.join(working_dir,'Verification')
net.animate_last_sim(dest = path+'verification_run_example.gif')
df:pd.DataFrame = net.parse_event_history()

print(df.to_latex)



# net1, counts1,sd, t_peak1, peak_height1, equilib_flag1, durchseuchung1 = simple_experiment(n,p,p_i, mc_iterations, max_t, force_recompute=True, path=path, mode = 'tracing', clustering=3)

# net1.animate_last_sim()

# print(net1.clustering())

# net1.plot_timeseries(counts1, sd=sd)

plt.show()



