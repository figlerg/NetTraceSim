# just checking whether the refactoring breaks anything
import pickle
import os
from net import Net

dirname_parent = os.path.dirname(__file__)
dirname = os.path.join(dirname_parent, 'Testcases_refactoring')

# old results
with open(os.path.join(dirname,'oldtest'),'rb') as f:
    old_results = pickle.load(f)

# new results
with open(os.path.join(dirname,'newtest'),'rb') as f:
    new_results = pickle.load(f)

net1, counts1, t_peak1, peak_height1, equilib_flag1, durchseuchung1 = old_results
net2, counts2, t_peak2, peak_height2, equilib_flag2, durchseuchung2 = new_results

print(counts2==counts1)
net1:Net
net2:Net

net1.plot_timeseries(counts1, save=os.path.join(dirname,'ts1.png'))
net2.plot_timeseries(counts2, save=os.path.join(dirname,'ts2.png'))

net1.animate_last_sim(dest=os.path.join(dirname,'vid1.mp4'))
net2.animate_last_sim(dest=os.path.join(dirname,'vid2.mp4'))