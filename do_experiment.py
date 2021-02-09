import os
import pickle
from net import Net
from montecarlo import monte_carlo
from globals import *




def simple_experiment(n,p,mc_iterations, max_t, mode = None, force_recompute = False, path = None):
    # this creates the net, runs monte carlo on it and saves the resulting timeseries plot, as well as pickles for net and counts

    if path:
        dirname = path
    else:
        dirname_parent = os.path.dirname(__file__)
        dirname = os.path.join(dirname_parent, 'Experiments')

    tag = "n{0}_p{1}_mc{2}".format(n,str(p), mc_iterations)
    if mode:
        tag += '_' + mode

    # TODO separate these try/except clauses. this way, only the files that are missing can be generated
    if force_recompute:
        # if false, it looks at saved experiments and reuses those
        net = Net(n=n, p=p, seed=123, max_t=max_t)
        counts = monte_carlo(net, 10, mode=mode)
        with open(os.path.join(dirname,tag+'_net.p'),'wb') as f:
            pickle.dump(net, f)
        with open(os.path.join(dirname,tag+'_counts.p'),'wb') as f:
            pickle.dump(counts,f)

        net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    else:
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
            counts = monte_carlo(net, 10, mode=mode)
            with open(os.path.join(dirname,tag+'_net.p'),'wb') as f:
                pickle.dump(net, f)
            with open(os.path.join(dirname,tag+'_counts.p'),'wb') as f:
                pickle.dump(counts,f)

            net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    exposed = counts[EXP_STATE,:]
    infected = counts[INF_STATE,:]
    ep_curve = exposed + infected

    # compute when the peak happens and what the number of infected is then
    t_peak = np.argmax(ep_curve,axis=0)
    peak_height = ep_curve[t_peak]

    # compute the ratio of all exposed people at end of sim to the number of indiv.
    # (also check heuristically whether an equilibrium has been reached

    recovered = counts[REC_STATE,:]
    virus_contacts = ep_curve + recovered

    sensitivity = max(1, n/100) # increasing divisor makes this more sensitive
    equilib_flag = abs(virus_contacts[-1] - virus_contacts[-2]) < sensitivity # just a heuristic, see whether roc is low

    durchseuchung = virus_contacts[-1] / n

    return (net, counts, t_peak, peak_height, equilib_flag, durchseuchung)


def vary_p(n, mc_iterations, max_t, mode = None, force_recompute = False, path = None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, durchseuchung

    for p in np.linspace(0,1, endpoint=True, num=10):
        net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations,max_t,mode, path = path)


