import matplotlib.pyplot as plt

from net import Net
import pickle
from do_experiment import *
import networkx as nx
import numpy as np
import pandas as pd
import random
import os

# 1st experiment:
#   1) set up a base case
#   2) simply create monte carlo time series that show the epidemic curves
#   3) compare between the three models


res = 5
n = 500
p = 10/500
p_i = 0.5
mc_iterations = 100
max_t = 400

working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

net1, counts1,sd1, t_peak1, peak_height1, equilib_flag1, durchseuchung1 = \
    simple_experiment(n,p,p_i, mc_iterations, max_t, force_recompute=False,
                      path=path, mode = None)
net2, counts2,sd2, t_peak2, peak_height2, equilib_flag2, durchseuchung2 = \
    simple_experiment(n,p,p_i, mc_iterations, max_t, force_recompute=False,
                      path=path, mode = 'quarantine')
net3, counts3,sd3, t_peak3, peak_height3, equilib_flag3, durchseuchung3 = \
    simple_experiment(n,p,p_i, mc_iterations, max_t, force_recompute=False,
                      path=path, mode = 'tracing')

peak_times = [t_peak1,t_peak2,t_peak3]
peak_heights = [peak_height1,peak_height2,peak_height3]
period_prevalences = [durchseuchung1,durchseuchung2,durchseuchung3]

d = {'Peak time':peak_times, 'Peak prevalence':peak_heights, 'Fraction of affected':period_prevalences}

frame = pd.DataFrame(data=d, index=['Vanilla', 'Quarantine','Tracing'])
latexstr = frame.to_latex()

print(frame)
print(latexstr)


fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,squeeze=True,figsize=(14, 3.5))

net1.plot_timeseries(counts1, existing_ax=ax1)
net2.plot_timeseries(counts2, existing_ax=ax2)
net3.plot_timeseries(counts3, existing_ax=ax3)

ax1.set_title('Vanilla Model')
ax2.set_title('Quarantine')
ax3.set_title('Tracing')

plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])

working_dir = os.getcwd()
path = os.path.join(working_dir,'Pics')


# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#                     hspace = 0, wspace = 0)
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(os.path.join(path,'base_comp'),bbox_inches = 'tight')