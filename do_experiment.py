import os
import pickle
from net import Net
from montecarlo import monte_carlo




def simple_experiment(n,p,mc_iterations, max_t):
    # this creates the net, runs monte carlo on it and saves the resulting timeseries plot, as well as pickles for net and counts

    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments')

    tag = "n{0}_p{1}_mc{2}".format(n,str(p), mc_iterations)
    try:
        with open(os.path.join(dirname,tag+"_counts.p"), 'rb') as f:
            counts = pickle.load(f)
        with open(os.path.join(dirname,tag+"_net.p"), 'rb') as f:
            net = pickle.load(f)
        with open(os.path.join(dirname,tag+"_vis.png"), 'rb') as f:
            pass

        print('Experiment results have been loaded from history.')


    except FileNotFoundError:
        net = Net(n=n, p=p, seed=123, max_t=max_t)
        counts = monte_carlo(net, 10)
        with open(os.path.join(dirname,tag+'_net.p'),'wb') as f:
            pickle.dump(net, f)
        with open(os.path.join(dirname,tag+'_counts.p'),'wb') as f:
            pickle.dump(counts,f)

        net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    return (net, counts)