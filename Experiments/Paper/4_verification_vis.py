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
name = os.path.join(path,'verification_run_example.mp4')
net.animate_last_sim(dest=name)
df:pd.DataFrame = net.parse_event_history()

name2 = os.path.join(path,'verification_run_example.txt')
with open(name2,'w') as f:
    f.write(df.to_latex())




