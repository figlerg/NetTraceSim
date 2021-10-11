# code is in net.py and montecarlo.py
import matplotlib.pyplot as plt

from net import Net
import pickle
from do_experiment import simple_experiment
import networkx as nx
import numpy as np
import random
import os



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
n = 1000
p = 0.03
p_i = 0.5
mc_iterations = 100
max_t = 200

working_dir = os.getcwd()
path = working_dir.join(['Experiments',])


if __name__ == '__main__':

    net1, counts1,sd, t_peak1, peak_height1, equilib_flag1, durchseuchung1,achieved_clustering, achieved_disp = simple_experiment(
        n, p, p_i, mc_iterations, max_t, mode='tracing', force_recompute=False, path=path, clustering=0.15)

    net2 = Net(n,p,p_i, max_t,seed=0, clustering_target=None, dispersion_target=None)


    print('Clustering- altered:{}, normal:{}'.format(net1.clustering(), net2.clustering()))
    print('Dispersion- altered:{}, normal:{}'.format(net1.dispersion(), net2.dispersion()))

    # net1.animate_last_sim()

    # print(net1.clustering())

    net1.plot_timeseries(counts1, sd=sd)

    plt.show()



