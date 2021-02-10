import os
import pickle
from net import Net
from globals import *
import matplotlib.pyplot as plt



# pickling disabled for now, uncomment plot lines for that
def simple_experiment(n,p,p_i,mc_iterations, max_t, mode = None, force_recompute = False, path = None, clustering:float = None):
    # this creates the net, runs monte carlo on it and saves the resulting timeseries plot, as well as pickles for net and counts
    # NOTE: SEED IS CONSTANT HERE!

    if path:
        dirname = path
    else:
        dirname_parent = os.path.dirname(__file__)
        dirname = os.path.join(dirname_parent, 'Experiments')

    tag = "n{0}_p{1}_mc{2}".format(n,str(p), mc_iterations)
    if mode:
        tag += '_' + mode
    if clustering:
        tag += '_clustering{}'.format(clustering)

    # TODO separate these try/except clauses. this way, only the files that are missing can be generated
    if force_recompute:
        # if false, it looks at saved experiments and reuses those
        net = Net(n=n, p=p,p_i=p_i , seed=123, max_t=max_t, clustering_target = clustering)
        counts = net.monte_carlo(mc_iterations, mode=mode)
        with open(os.path.join(dirname,tag+'_net.p'),'wb') as f:
            pickle.dump(net, f)
        with open(os.path.join(dirname,tag+'_counts.p'),'wb') as f:
            pickle.dump(counts,f)

        # net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    else:
        try:
            with open(os.path.join(dirname,tag+"_counts.p"), 'rb') as f:
                counts = pickle.load(f)
            with open(os.path.join(dirname,tag+"_net.p"), 'rb') as f:
                net = pickle.load(f)
            # with open(os.path.join(dirname,tag+"_vis.png"), 'rb') as f:
            #     pass

            print('Experiment results have been loaded from history.')


        except FileNotFoundError:
            net = Net(n=n, p=p, p_i=p_i, seed=123, max_t=max_t, clustering_target=clustering)

            counts = net.monte_carlo(mc_iterations, mode=mode)
            with open(os.path.join(dirname,tag+'_net.p'),'wb') as f:
                pickle.dump(net, f)
            with open(os.path.join(dirname,tag+'_counts.p'),'wb') as f:
                pickle.dump(counts,f)

            # net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

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


def vary_p(res, n, p_i, mc_iterations, max_t, mode = None, force_recompute = False, path = None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, durchseuchung

    # res parameter defines how many points on [0,1] are used
    res = res-1 # silly, but for me it makes more sense to create n values for res = n (linspace creates n+1 with endpoint)
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    # flags = np.ndarray(res)
    durchseuchungen = np.ndarray(res)

    ps = np.linspace(0,1, endpoint=True, num=res)

    for i,p in enumerate(ps):
        net, counts, t_peak, peak_height, equilib_flag, durchseuchung = \
            simple_experiment(n,p,p_i,mc_iterations,max_t,mode, path = path, force_recompute = force_recompute)

        peak_times[i] = t_peak
        peak_heights[i] = peak_height
        durchseuchungen[i] = durchseuchung

    fig, axes = plt.subplots(3,1, sharex=True, figsize=(16*1.5,9*1.5))
    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes


    ax1.plot(ps,peak_times)
    # ax1.set_xlabel('p')
    ax1.set_ylabel('peak-time')

    ax2.plot(ps,peak_heights)
    # ax2.set_xlabel('p')
    ax2.set_ylabel('peak-height')

    ax3.plot(ps,durchseuchungen)
    # ax3.set_xlabel('p')
    ax3.set_ylabel('percentage of affected')
    ax3.set_xlabel('p')
    ax3.set_xticks(ps)
    if mode:
        fig.savefig(os.path.join(path, 'pvaried_n{}_pi{}_{}'.format(n, p_i, mode)+ '.png'))
    else:
        fig.savefig(os.path.join(path, 'pvaried_n{}_pi{}'.format(n, p_i) + '.png'))


def vary_p_i(res, n, p, mc_iterations, max_t, mode = None, force_recompute = False, path = None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, durchseuchung

    # res parameter defines how many points on [0,1] are used
    res = res-1 # silly, but for me it makes more sense to create n values for res = n (linspace creates n+1 with endpoint)
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    # flags = np.ndarray(res)
    durchseuchungen = np.ndarray(res)

    p_is = np.linspace(0,1, endpoint=True, num=res)

    for i,p_inf in enumerate(p_is):
        net, counts, t_peak, peak_height, equilib_flag, durchseuchung = \
            simple_experiment(n,p,p_inf,mc_iterations,max_t,mode, path = path, force_recompute = force_recompute)
        # TODO seed inside simple_experiment is constant, think about whether that's ok!

        peak_times[i] = t_peak
        peak_heights[i] = peak_height
        durchseuchungen[i] = durchseuchung

    fig, axes = plt.subplots(3,1, sharex=True, figsize=(16*1.5,9*1.5))
    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes


    ax1.plot(p_is,peak_times)
    # ax1.set_xlabel('p')
    ax1.set_ylabel('peak-times')

    ax2.plot(p_is,peak_heights)
    # ax2.set_xlabel('p')
    ax2.set_ylabel('peak-height')

    ax3.plot(p_is,durchseuchungen)
    # ax3.set_xlabel('p')
    ax3.set_ylabel('percentage of affected')
    ax3.set_xlabel('infection probability')
    ax3.set_xticks(p_is)
    # plt.show()
    if mode:
        fig.savefig(os.path.join(path, 'pivaried_n{}_p{}_{}'.format(n, p, mode)+ '.png'))
    else:
        fig.savefig(os.path.join(path, 'pivaried_n{}_p{}'.format(n, p)+ '.png'))



if __name__ == '__main__':
    res = 20
    n = 500
    p = 0.1
    p_i = 0.5
    mc_iterations = 50
    max_t = 200
    path = path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\vary_params'


    vary_p(res=res,n=n,p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, path = path)
    vary_p(res=res,n=n,p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='quarantine', path = path)
    vary_p(res=res,n=n,p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='tracing', path = path)

    vary_p_i(res=res,n=n,p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, path = path)
    vary_p_i(res=res,n=n,p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='quarantine', path = path)
    vary_p_i(res=res,n=n,p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='tracing', path = path)

    # vary_p(res=3,n=100,p_i=0.5, mc_iterations=1, max_t=20 path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\vary_params')