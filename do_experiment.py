import hashlib
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl

from globals import *
from net import Net
from tqdm import tqdm
import cycler

import random
random.seed(12345)

#Direct input
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"
# plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 10,
          # 'font.family' : 'lmodern',
          }
plt.rcParams.update(params)

columwidth = 251.8/72.27 # 251.80688[pt] / 72.27[pt/inch]


def estimateQuotientCI(ax,xvalues,mean1,sd1,mean2,sd2,color,mccount,p=95):
    iters = 2000
    lowers = list()
    uppers = list()
    percs = [(100-p)/2,100-(100-p)/2]

    """
    Monte Carlo mean 1/N*sum(X_i) implies:
    V(1/N*sum(X_i))=1/(N^2)*sum(V(X_i))=1/(N^2)*N*V(X)=V(X)/N
    => Variance of monte carlo mean is 1/N times variance of single model result
    => SD is 1/sqrt(N) times SD of model result
    """
    sd11 = sd1/(mccount**0.5)
    sd21 = sd2/(mccount**0.5)
    for m1,s1,m2,s2 in zip(mean1,sd11,mean2,sd21):
        quotients = list()
        for i in range(iters):
            """
            since (sum(X_i)-mu)/(sqrt(N)*sigma) converges towards Normal(0,1) we may
            assume  1/N*sum(X_i) approx ~ Normal(mu,sigma/sqrt(N))
            """
            denom = random.normalvariate(m2,s2)
            if denom<=0: #truncate normal dist - negative values dont make sense
                continue
            nom = random.normalvariate(m1,s1)
            if nom<0: #truncate normal dist - negative values dont make sense
                continue
            quotients.append(nom/denom)
        ps = np.percentile(quotients,percs)
        lowers.append(ps[0])
        uppers.append(ps[1])
    ax.fill_between(xvalues,lowers,uppers,color=color,alpha=0.2,zorder=-1)

# pickling disabled for now, uncomment plot lines for that
def simple_experiment_old(n, p, p_i, mc_iterations, max_t, seed=0, mode=None, force_recompute=False, path=None,
                      clustering: float = None, dispersion=None):
    # this creates the net, runs monte carlo on it and saves the resulting timeseries plot, as well as pickles for net and counts

    assert not (dispersion and clustering), "Cannot set a dispersion target and " \
                                            "a clustering target at the same time"
    if dispersion:
        chosen_epsilon = epsilon_disp
    else:
        chosen_epsilon = epsilon_clustering

    if path:
        dirname = path
    else:
        dirname_parent = os.path.dirname(__file__)
        dirname = os.path.join(dirname_parent, 'Experiments')

    # the cache is now tagged with a hash from all important parameters instead of the above.
    # Any change to the model parameters will certainly trigger a recompute now
    id_params = (
        n, p, p_i, mc_iterations, max_t, seed, mode, clustering, dispersion, t_i, t_c, t_r, t_d, t_t, p_q, p_t,
        quarantine_time, resolution, chosen_epsilon)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    # disables loading pickled results
    if force_recompute:
        # if false, it looks at saved experiments and reuses those
        net = Net(n=n, p=p, p_i=p_i, max_t=max_t, seed=seed, clustering_target=clustering, dispersion_target=dispersion)
        counts, sd, achieved_clustering, achieved_disp = net.monte_carlo(mc_iterations, mode=mode)
        with open(os.path.join(dirname, tag + '_net.p'), 'wb') as f:
            pickle.dump((net, achieved_clustering, achieved_disp), f)
        with open(os.path.join(dirname, tag + '_counts.p'), 'wb') as f:
            pickle.dump((counts, sd), f)


    else:
        try:
            with open(os.path.join(dirname, tag + "_counts.p"), 'rb') as f:
                counts, sd = pickle.load(f)
            with open(os.path.join(dirname, tag + "_net.p"), 'rb') as f:
                net, achieved_clustering, achieved_disp = pickle.load(f)
            print('Experiment results have been loaded from history.')

        except FileNotFoundError:
            net = Net(n=n, p=p, p_i=p_i, max_t=max_t, seed=seed, clustering_target=clustering,
                      dispersion_target=dispersion)

            counts, sd, achieved_clustering, achieved_disp = net.monte_carlo(mc_iterations, mode=mode)
            with open(os.path.join(dirname, tag + '_net.p'), 'wb') as f:
                pickle.dump((net, achieved_clustering, achieved_disp), f)
            with open(os.path.join(dirname, tag + '_counts.p'), 'wb') as f:
                pickle.dump((counts, sd), f)

    exposed = counts[EXP_STATE, :]
    infected = counts[INF_STATE, :]
    ep_curve = exposed + infected

    # compute when the peak happens and what the ratio of infected is then
    t_peak = np.argmax(ep_curve, axis=0)
    peak_height = ep_curve[t_peak] / n

    # compute the ratio of all exposed people at end of sim to the number of indiv.
    # (also check heuristically whether an equilibrium has been reached

    recovered = counts[REC_STATE, :]
    virus_contacts = ep_curve + recovered

    sensitivity = max(1, n / 100)  # increasing divisor makes this more sensitive
    equilib_flag = abs(
        virus_contacts[-1] - virus_contacts[-2]) < sensitivity  # just a heuristic, see whether roc is low

    period_prevalence = virus_contacts[-1] / n
    return net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence, achieved_clustering, achieved_disp

from do_experiment_parallel import simple_experiment # this is the new, parallel version of the above function. By Martin!


def vary_p(res, n, p_i, mc_iterations, max_t, interval=(0, 1), seed=0, mode=None, force_recompute=False, path=None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, period_prevalence

    peak_times = np.ndarray(res)
    mean_peak_heights = np.ndarray(res)
    mean_period_prevalences = np.ndarray(res)
    sd_peak_heights = np.ndarray(res)
    sd_period_prevalences = np.ndarray(res)

    ps = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    for i, p in enumerate(ps):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , clustering, dispersion = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i, mode=mode,
                              force_recompute=force_recompute, path=path)

        peak_times[i] = t_peak
        mean_peak_heights[i] = mean_peak
        sd_peak_heights[i] = sd_peak
        mean_period_prevalences[i] = mean_prevalence
        sd_period_prevalences[i] = sd_prevalence

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16 * 1.5, 9 * 1.5))

    ax1, ax2, ax3 = axes

    if mode:
        ax1.set_title(mode)
    else:
        ax1.set_title('vanilla')

    ax1.plot(ps, peak_times)
    # ax1.set_xlabel('p')
    ax1.set_ylabel('Peak time')

    ax2.plot(ps, mean_peak_heights)
    # ax2.set_xlabel('p')
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(ps, mean_period_prevalences)
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('p')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(ps[1:-2], minor=True)
    ax3.set_xticks([interval[0], interval[1]])

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        # bottom=False,      # ticks along the bottom edge are off
        # top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    if mode:
        fig.savefig(os.path.join(path, 'pvaried_n{}_p{}_{}'.format(
            n, str(interval[0]) + 'to' + str(interval[1]), mode) + '.png'))
    else:
        fig.savefig(os.path.join(path, 'pvaried_n{}_p{}'.format(
            n, str(interval[0]) + 'to' + str(interval[1])) + '.png'))


def vary_p_plot_cache(res, n, p_i, mc_iterations, max_t, interval=(0, 1), seed=0, force_recompute=False, path=None):
    # utility function that loads all the pickles (or runs them first) and plots the three scenarios
    # is a modified copy of vary_p !

    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    period_prevalences = np.ndarray(res)
    peak_times_q = np.ndarray(res)
    peak_heights_q = np.ndarray(res)
    period_prevalences_q = np.ndarray(res)
    peak_times_t = np.ndarray(res)
    peak_heights_t = np.ndarray(res)
    period_prevalences_t = np.ndarray(res)

    peak_heights_sd = np.ndarray(res)
    peak_heights_q_sd = np.ndarray(res)
    peak_heights_t_sd = np.ndarray(res)
    period_prevalences_sd = np.ndarray(res)
    period_prevalences_q_sd = np.ndarray(res)
    period_prevalences_t_sd = np.ndarray(res)

    ps = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    # all 3 modes
    for i, p in tqdm(enumerate(ps),total=res, desc='Vanilla'):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + res, mode=None,
                              force_recompute=force_recompute, path=path)

        peak_times[i] = t_peak
        peak_heights[i] = mean_peak
        peak_heights_sd[i] = sd_peak
        period_prevalences[i] = mean_prevalence
        period_prevalences_sd[i] = sd_prevalence

    for i, p in tqdm(enumerate(ps),total=res, desc='Quarantine'):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + 2 * res, mode='quarantine',
                              force_recompute=force_recompute,
                              path=path)

        peak_times_q[i] = t_peak
        peak_heights_q[i] = mean_peak
        peak_heights_q_sd[i] = sd_peak
        period_prevalences_q[i] = mean_prevalence
        period_prevalences_q_sd[i] = sd_prevalence

    for i, p in tqdm(enumerate(ps),total=res, desc='Tracing'):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + 3 * res, mode='tracing',
                              force_recompute=force_recompute,
                              path=path)

        peak_times_t[i] = t_peak
        peak_heights_t[i] = mean_peak
        peak_heights_t_sd[i] = sd_peak
        period_prevalences_t[i] = mean_prevalence
        period_prevalences_t_sd[i] = sd_prevalence

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14 / 16 * 9))
    ax1, ax2, ax3 = axes

    ax1.plot(ps, peak_times, ps, peak_times_q, ps, peak_times_t)
    ax1.set_ylabel('Peak time')

    ax2.plot(ps, peak_heights, ps, peak_heights_q, ps, peak_heights_t)
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(ps, period_prevalences, ps, period_prevalences_q, ps, period_prevalences_t)
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('p')
    ax3.set_xticks(ps[1:-2], minor=True)
    ax3.set_xticks([interval[0], interval[1]])

    plt.legend(['Vanilla', 'Quarantine', 'Tracing'])

    plt.tick_params(
        axis='x',
        which='minor',
        # bottom=False,      # ticks along the bottom edge are off
        # top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    parent = os.path.dirname(path)
    fig.savefig(os.path.join(parent, 'Pics', 'pvaried_n{}_mc{}_{}'.format(n, mc_iterations, 'comp') + '.png'),
                bbox_inches='tight')


# this feels pretty uninteresting:
def vary_p_i(res, n, p, mc_iterations, max_t, seed=0, mode=None, force_recompute=False, path=None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, period_prevalence

    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    peak_heights_sd = np.ndarray(res)
    # flags = np.ndarray(res)
    period_prevalences = np.ndarray(res)
    period_prevalences_sd = np.ndarray(res)

    p_is = np.linspace(0, 1, endpoint=True, num=res)

    for i, p_inf in enumerate(p_is):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_inf, mc_iterations, max_t, seed=seed + i, mode=mode,
                              force_recompute=force_recompute, path=path)
        # TODO seed inside simple_experiment is constant, think about whether that's ok!

        peak_times[i] = t_peak
        peak_heights[i] = mean_peak
        peak_heights_sd[i] = sd_peak
        period_prevalences[i] = mean_prevalence
        period_prevalences_sd[i] = sd_prevalence

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16 * 1.5, 9 * 1.5))
    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    ax1.plot(p_is, peak_times)
    # ax1.set_xlabel('p')
    ax1.set_ylabel('peak-times')

    ax2.plot(p_is, peak_heights)
    # ax2.set_xlabel('p')
    ax2.set_ylabel('peak-height')

    ax3.plot(p_is, period_prevalences)
    # ax3.set_xlabel('p')
    ax3.set_ylabel('percentage of affected')
    ax3.set_xlabel('infection probability')
    ax3.set_xticks(p_is)
    # plt.show()
    if mode:
        fig.savefig(os.path.join(path, 'pivaried_n{}_p{}_{}'.format(n, p, mode) + '.png'))
    else:
        fig.savefig(os.path.join(path, 'pivaried_n{}_p{}'.format(n, p) + '.png'))

def vary_C(res, n, p, p_i, mc_iterations, max_t, interval=None, seed=0, mode=None, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5 * p, 10 * p)

    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    peak_heights_sd = np.ndarray(res)
    period_prevalences = np.ndarray(res)
    period_prevalences_sd = np.ndarray(res)

    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    unsuccessful_flag = []
    for i, C in tqdm(enumerate(Cs), total=res):
        try:
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i, mode=mode,
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times[i] = t_peak
            peak_heights[i] = mean_peak
            peak_heights_sd[i] = sd_peak
            period_prevalences[i] = mean_prevalence
            period_prevalences_sd[i] = sd_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flag.append(i)
            peak_times[i] = np.nan
            peak_heights[i] = np.nan
            peak_heights_sd[i] = np.nan
            period_prevalences[i] = np.nan
            period_prevalences_sd[i] = np.nan

    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments', 'Paper', 'Cache')

    id_params = (
        n, p, p_i, mc_iterations, max_t, seed, mode, interval, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time,
        resolution,
        epsilon_disp, 'disp')
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname, tag + '_metrics.p'), 'wb') as f:
        out = [Cs, unsuccessful_flag, peak_times, peak_heights, period_prevalences]

        pickle.dump(out, f)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14 / 16 * 9))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    colordict = {'vanilla': 'C0', 'quarantine': 'C1', 'tracing': 'C2'}

    if mode:
        ax1.set_title(mode)
    else:
        ax1.set_title('Vanilla')

    ax1.plot(Cs, peak_times, colordict[mode])
    ax1.set_ylabel('Peak time')

    ax2.plot(Cs, peak_heights, colordict[mode])
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(Cs, period_prevalences, colordict[mode])
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Cs[1:-1], minor=True)
    ax3.set_xticks([interval[0], interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    if mode:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent, 'Pics', 'Cvaried_n{}_p{}_{}'.format(
            n, str(interval[0]) + 'to' + str(interval[1]), mode) + '.png'), bbox_inches='tight')
    else:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent, 'Pics', 'Cvaried_n{}_C{}'.format(
            n, str(interval[0]) + 'to' + str(interval[1])) + '.png'), bbox_inches='tight')

    return out  # Cs, unsuccessful_flags, times, peaks, period_prev


def vary_disp(res, n, p, p_i, mc_iterations, max_t, interval=None, seed=0, mode=None, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5 * p, 10 * p)

    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    peak_heights_sd = np.ndarray(res)
    period_prevalences = np.ndarray(res)
    period_prevalences_sd = np.ndarray(res)

    Ds = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    unsuccessful_flag = []
    for i, D in tqdm(enumerate(Ds),total=res, desc='Varying dispersion values'):
        try:
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i, mode=mode,
                                  force_recompute=force_recompute,
                                  path=path, dispersion=D)
            peak_times[i] = t_peak
            peak_heights[i] = mean_peak
            peak_heights_sd[i] = sd_peak
            period_prevalences[i] = mean_prevalence
            period_prevalences_sd[i] = sd_prevalence

            print('last_disp{}, target_disp{}'.format(net.final_dispersion, D))

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Dispersion target not reached')

            unsuccessful_flag.append(i)
            peak_times[i] = np.nan
            peak_heights[i] = np.nan
            peak_heights_sd[i] = np.nan
            period_prevalences[i] = np.nan
            period_prevalences_sd[i] = np.nan

    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments', 'Paper', 'Cache')

    id_params = (
        n, p, p_i, mc_iterations, max_t, mode, seed, interval, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time,
        resolution,
        epsilon_disp)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname, tag + '_metrics.p'), 'wb') as f:
        out = [Ds, unsuccessful_flag, peak_times, peak_heights, period_prevalences]

        pickle.dump(out, f)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14 / 16 * 9))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    colordict = {'vanilla': 'C0', 'quarantine': 'C1', 'tracing': 'C2'}

    if mode:
        ax1.set_title(mode)
    else:
        ax1.set_title('Vanilla')

    ax1.plot(Ds, peak_times, colordict[mode])
    ax1.set_ylabel('Peak time')

    ax2.plot(Ds, peak_heights, colordict[mode])
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(Ds, period_prevalences, colordict[mode])
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Ds[1:-1], minor=True)
    ax3.set_xticks([interval[0], interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    if mode:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent, 'Pics', 'dispvaried_n{}_p{}_{}'.format(
            n, str(interval[0]) + 'to' + str(interval[1]), mode) + '.png'), bbox_inches='tight')
    else:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent, 'Pics', 'dispvaried_n{}_C{}'.format(
            n, str(interval[0]) + 'to' + str(interval[1])) + '.png'), bbox_inches='tight')

    return out  # Cs, unsuccessful_flags, times, peaks, period_prev


def vary_C_comp(res, n, p, p_i, mc_iterations, max_t, interval=None, seed=0, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5 * p, 10 * p)

    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    peak_times_1 = np.ndarray(res)
    peak_heights_1 = np.ndarray(res)
    peak_heights_sd_1 = np.ndarray(res)
    period_prevalences_1 = np.ndarray(res)
    period_prevalences_sd_1 = np.ndarray(res)

    unsuccessful_flags_1 = []
    for i, C in tqdm(enumerate(Cs), total=res,desc='Vanilla'):
        try:
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i, mode='vanilla',
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_1[i] = t_peak
            peak_heights_1[i] = mean_peak
            peak_heights_sd_1[i] = sd_peak
            period_prevalences_1[i] = mean_prevalence
            period_prevalences_sd_1[i] = sd_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_1.append(i)
            peak_times_1[i] = np.nan
            peak_heights_1[i] = np.nan
            peak_heights_sd_1[i] = np.nan
            period_prevalences_1[i] = np.nan
            period_prevalences_sd_1[i] = np.nan

    peak_times_2 = np.ndarray(res)
    peak_heights_2 = np.ndarray(res)
    peak_heights_sd_2 = np.ndarray(res)
    period_prevalences_2 = np.ndarray(res)
    period_prevalences_sd_2 = np.ndarray(res)
    unsuccessful_flags_2 = []
    for i, C in tqdm(enumerate(Cs), total=res,desc='Quarantine'):
        try:
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + res, mode='quarantine',
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_2[i] = t_peak
            peak_heights_2[i] = mean_peak
            peak_heights_sd_2[i] = sd_peak
            period_prevalences_2[i] = mean_prevalence
            period_prevalences_sd_2[i] = sd_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_2.append(i)
            peak_times_2[i] = np.nan
            peak_heights_2[i] = np.nan
            peak_heights_sd_2[i] = np.nan
            period_prevalences_2[i] = np.nan
            period_prevalences_sd_2[i] = np.nan

    peak_times_3 = np.ndarray(res)
    peak_heights_3 = np.ndarray(res)
    peak_heights_sd_3 = np.ndarray(res)
    period_prevalences_3 = np.ndarray(res)
    period_prevalences_sd_3 = np.ndarray(res)
    unsuccessful_flags_3 = []
    for i, C in tqdm(enumerate(Cs), total=res, desc='Tracing'):
        try:

            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag, achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + 2 * res, mode='tracing',
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_3[i] = t_peak
            peak_heights_3[i] = mean_peak
            peak_heights_sd_3[i] = sd_peak
            period_prevalences_3[i] = mean_prevalence
            period_prevalences_sd_3[i] = sd_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_3.append(i)
            peak_times_3[i] = np.nan
            peak_heights_3[i] = np.nan
            peak_heights_sd_3[i] = np.nan
            period_prevalences_3[i] = np.nan
            period_prevalences_sd_3[i] = np.nan

    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments', 'Paper', 'Cache')

    id_params = (
        n, p, p_i, mc_iterations, max_t, interval, seed, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution,
        epsilon_clustering)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname, tag + '_metrics.p'), 'wb') as f:
        out = [Cs, unsuccessful_flags_1, peak_times_1, peak_heights_1, period_prevalences_1,
               Cs, unsuccessful_flags_2, peak_times_2, peak_heights_2, period_prevalences_2,
               Cs, unsuccessful_flags_3, peak_times_3, peak_heights_3, period_prevalences_3]

        pickle.dump(out, f)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14 / 16 * 9))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    ax1.plot(Cs, peak_times_1, Cs, peak_times_2, Cs, peak_times_3)
    ax1.set_ylabel('Peak time')

    ax2.plot(Cs, peak_heights_1, Cs, peak_heights_2, Cs, peak_heights_3)
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(Cs, period_prevalences_1, Cs, period_prevalences_2, Cs, period_prevalences_3)
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Cs[1:-1], minor=True)
    ax3.set_xticks([interval[0], interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])
    plt.legend(['Vanilla', 'Quarantine', 'Tracing'])

    parent = os.path.dirname(path)
    fig.savefig(os.path.join(parent, 'Pics', 'Cvaried_n{}_C{}_comp'.format(
        n, str(interval[0]) + 'to' + str(interval[1])) + '.png'), bbox_inches='tight')

    return out  # Cs, unsuccessful_flags, times, peaks, period_prev


def vary_C_comp_corrected(res, n, p, p_i, mc_iterations, max_t, interval=None, seed=0, force_recompute=False,
                          path=None):
    # BROKEN! Since martin's commit?

    # measure effect of clustering coeff on tracing effectiveness. Here we scale according to the vanilla outcome

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5 * p, 10 * p)

    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    # the following two variables save the actual values that were achieved by the heuristic.
    # In theory, these should be approximately the same in each net
    achieved_clusterings = np.zeros((3, res))
    achieved_disps = np.zeros((3, res))

    # vanilla
    peak_times_1 = np.ndarray(res)
    peak_heights_1 = np.ndarray(res)
    peak_heights_sd_1 = np.ndarray(res)
    period_prevalences_1 = np.ndarray(res)
    period_prevalences_sd_1 = np.ndarray(res)
    unsuccessful_flags_1 = []
    for i, C in tqdm(enumerate(Cs), total=res,desc='Vanilla'):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i, mode='vanilla',
                              force_recompute=force_recompute,
                              path=path, clustering=C)

        assert equilib_flag, 'Sim not complete?'

        peak_times_1[i] = t_peak
        peak_heights_1[i] = mean_peak
        peak_heights_sd_1[i] = sd_peak
        period_prevalences_1[i] = mean_prevalence
        period_prevalences_sd_1[i] = sd_prevalence

        achieved_clusterings[0, i] = achieved_clustering
        achieved_disps[0, i] = achieved_disp


        # exposed = counts[EXP_STATE, :]
        # infected = counts[INF_STATE, :]
        # ep_curve = exposed + infected
        #
        # exposed_sd = sd[EXP_STATE, :]
        # infected_sd = sd[INF_STATE, :]
        # ep_curve_sd = exposed_sd + infected_sd
        #
        # # these are the point prevalence +- sd
        # upper_alpha = (ep_curve[t_peak] + ep_curve_sd[t_peak])/n
        # lower_alpha = (ep_curve[t_peak] - ep_curve_sd[t_peak])/n
        #
        # recovered = counts[REC_STATE, :]
        # recovered_sd = sd[REC_STATE, :]
        #
        #
        # upper_beta = recovered[-1]-recovered_sd/n


    # quarantine
    peak_times_2 = np.ndarray(res)
    peak_heights_2 = np.ndarray(res)
    peak_heights_sd_2 = np.ndarray(res)
    period_prevalences_2 = np.ndarray(res)
    period_prevalences_sd_2 = np.ndarray(res)
    unsuccessful_flags_2 = []
    for i, C in tqdm(enumerate(Cs), total=res,desc='Quarantine'):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + res, mode='quarantine',
                              force_recompute=force_recompute,
                              path=path, clustering=C)

        assert equilib_flag, 'Sim not complete?'

        peak_times_2[i] = t_peak
        peak_heights_2[i] = mean_peak / peak_heights_1[i]
        peak_heights_sd_2[i] = sd_peak
        period_prevalences_2[i] = mean_prevalence / period_prevalences_1[i]
        period_prevalences_sd_2[i] = sd_prevalence

        achieved_clusterings[1, i] = achieved_clustering
        achieved_disps[1, i] = achieved_disp




    # tracing
    peak_times_3 = np.ndarray(res)
    peak_heights_3 = np.ndarray(res)
    peak_heights_sd_3 = np.ndarray(res)
    period_prevalences_3 = np.ndarray(res)
    period_prevalences_sd_3 = np.ndarray(res)
    unsuccessful_flags_3 = []
    for i, C in tqdm(enumerate(Cs), total=res,desc='Tracing'):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, 2*mc_iterations, max_t, seed=seed + i + 2 * res, mode='tracing',
                              force_recompute=force_recompute,
                              path=path, clustering=C)

        assert equilib_flag, 'Sim not complete?'

        peak_times_3[i] = t_peak
        peak_heights_3[i] = mean_peak / peak_heights_1[i]
        peak_heights_sd_3[i] = sd_peak
        period_prevalences_3[i] = mean_prevalence / period_prevalences_1[i]
        period_prevalences_3_sd_2[i] = sd_prevalencea

        achieved_clusterings[2, i] = achieved_clustering
        achieved_disps[2, i] = achieved_disp

    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments', 'Paper', 'Cache')

    id_params = (
        n, p, p_i, mc_iterations, max_t, interval, seed, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution,
        epsilon_clustering)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname, tag + '_metrics_corrected.p'), 'wb') as f:
        out = [Cs, unsuccessful_flags_1, peak_times_1, peak_heights_1, period_prevalences_1,
               Cs, unsuccessful_flags_2, peak_times_2, peak_heights_2, period_prevalences_2,
               Cs, unsuccessful_flags_3, peak_times_3, peak_heights_3, period_prevalences_3,
               achieved_clusterings, achieved_disps]

        pickle.dump(out, f)

    # two modes for visualization
    show_both = False
    if show_both:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 14 / 16 * 9))

        # fig.subplots_adjust(wspace = 0.5)
        ax2, ax3 = axes

        # ax1.plot(Cs, peak_times_1,Cs, peak_times_2,Cs, peak_times_3)
        # ax1.set_ylabel('Peak time')

        ax2.plot(Cs, peak_heights_2, 'C1')
        ax2.plot(Cs, peak_heights_3, 'C2')
        ax2.set_ylabel('Scaled peak height')

        ax3.plot(Cs, period_prevalences_2, 'C1')
        ax3.plot(Cs, period_prevalences_3, 'C2')
        ax3.set_ylabel('Scaled period prevalence')
        ax3.set_xlabel('C(g)')
        # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
        ax3.set_xticks(Cs, minor=False)
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # ax3.set_xticks([interval[0], interval[1]])

        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='minor',      # both major and minor ticks are affected
        #     # bottom=False,      # ticks along the bottom edge are off
        #     # top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off

        ax_upper_axis = ax2.twiny()

        ax_upper_axis.set_xlim(ax3.get_xlim())
        ax_upper_axis.set_xticks(Cs)
        ax_upper_axis.set_xticklabels(["{:.3f}".format(a) for a in achieved_disps.mean(axis=0)])
        # ax_upper_axis.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_upper_axis.set_xlabel('D(g)')

        # plt.xticks([interval[0],interval[1]])
        ax3.legend(['Quarantine', 'Tracing'])

        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent, 'Pics', 'Cvaried_n{}_C{}_comp_corrected'.format(
            n, str(interval[0]) + 'to' + str(interval[1])) + '.png'), bbox_inches='tight')
    else:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 14 / 16 * 9))

        # fig.subplots_adjust(wspace = 0.5)
        ax2, ax3 = axes

        # ax1.plot(Cs, peak_times_1,Cs, peak_times_2,Cs, peak_times_3)
        # ax1.set_ylabel('Peak time')

        ax2.plot(Cs, peak_heights_3, 'C2')
        ax2.set_ylabel('Scaled peak height')

        ax3.plot(Cs, period_prevalences_3, 'C2')
        ax3.set_ylabel('Scaled period prevalence')
        ax3.set_xlabel('C(g)')
        ax3.set_xticks(Cs, minor=False)
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # ax3.set_xticks(Cs[1:-1], minor=True)
        # ax3.set_xticks([interval[0], interval[1]])
        # ax3.set_xticks(Cs, minor=True)

        ax_upper_axis = ax2.twiny()

        ax_upper_axis.set_xlim(ax3.get_xlim())
        ax_upper_axis.set_xticks(Cs)
        ax_upper_axis.set_xticklabels(["{:.3f}".format(a) for a in achieved_disps.mean(axis=0)])
        # ax_upper_axis.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_upper_axis.set_xlabel('D(g)')

        # plt.legend(['Quarantine', 'Tracing'])
        ax3.legend(['Tracing', ])

        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent, 'Pics', 'Cvaried_n{}_C{}_comp_corrected_tracing'.format(
            n, str(interval[0]) + 'to' + str(interval[1])) + '.png'), bbox_inches='tight')

    return out

def vary_C_pi_comp_corrected(res, n, p, p_is:tuple, mc_iterations, max_t, interval=None, seed=0, force_recompute=False,
                             path=None):
    # measure effect of clustering coeff on tracing effectiveness. Here we scale according to the vanilla outcome
    # Several values for infectvity p_i are used

    assert len(p_is) <= 3, 'Only three values for p_i should be selected for visibility'

    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)
    n_p_i = len(p_is)

    # the following two variables save the actual values that were achieved by the heuristic.
    # In theory, these should be approximately the same in each net
    achieved_clusterings = np.zeros((3*n_p_i, res))
    achieved_disps = np.zeros((3*n_p_i, res))

    # vanilla


    peak_times_1 = np.ndarray((res,n_p_i))
    peak_heights_1 = np.ndarray((res,n_p_i))
    peak_heights_sd_1 = np.ndarray((res,n_p_i))
    period_prevalences_1 = np.ndarray((res,n_p_i))
    period_prevalences_sd_1 = np.ndarray((res,n_p_i))
    for i, C in tqdm(enumerate(Cs), total=res,desc='Vanilla'):
        for j, p_inf in enumerate(p_is):
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_inf, mc_iterations, max_t, seed=j*156484+ seed + i, mode='vanilla',
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)

            assert equilib_flag, 'Sim not complete?'

            peak_times_1[i,j] = t_peak
            peak_heights_1[i,j] = mean_peak
            peak_heights_sd_1[i,j] = sd_peak
            period_prevalences_1[i,j] = mean_prevalence
            period_prevalences_sd_1[i,j] = sd_prevalence

            achieved_clusterings[j, i] = achieved_clustering
            achieved_disps[j, i] = achieved_disp



    # quarantine
    peak_times_2 = np.ndarray((res,n_p_i))
    peak_heights_2 = np.ndarray((res,n_p_i))
    peak_heights_sd_2 = np.ndarray((res,n_p_i))
    period_prevalences_2 = np.ndarray((res,n_p_i))
    period_prevalences_sd_2 = np.ndarray((res,n_p_i))
    for i, C in tqdm(enumerate(Cs), total=res,desc='Quarantine'):
        for j, p_inf in enumerate(p_is):
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_inf, mc_iterations, max_t, seed=j*84265+seed + i + res, mode='quarantine',
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)

            assert equilib_flag, 'Sim not complete?'

            peak_times_2[i,j] = t_peak
            peak_heights_2[i,j] = mean_peak
            peak_heights_sd_2[i,j] = sd_peak
            period_prevalences_2[i,j] = mean_prevalence
            period_prevalences_sd_2[i,j] = sd_prevalence

            achieved_clusterings[n_p_i+j, i] = achieved_clustering
            achieved_disps[n_p_i+j, i] = achieved_disp


    # tracing
    peak_times_3 = np.ndarray((res,n_p_i))
    peak_heights_3 = np.ndarray((res,n_p_i))
    peak_heights_sd_3 = np.ndarray((res,n_p_i))
    period_prevalences_3 = np.ndarray((res,n_p_i))
    period_prevalences_sd_3 = np.ndarray((res,n_p_i))
    for i, C in tqdm(enumerate(Cs), total=res,desc='Tracing'):
        for j, p_inf in enumerate(p_is):
            net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
                simple_experiment(n, p, p_inf, mc_iterations, max_t, seed=j*543513+seed + i + 2 * res, mode='tracing',
                                  force_recompute=force_recompute,
                                  path=path, clustering=C)

            assert equilib_flag, 'Sim not complete?'

            peak_times_3[i,j] = t_peak
            peak_heights_3[i,j] = mean_peak
            peak_heights_sd_3[i,j] = sd_peak
            period_prevalences_3[i,j] = mean_prevalence
            period_prevalences_sd_3[i,j] = sd_prevalence


            achieved_clusterings[2*n_p_i+j, i] = achieved_clustering
            achieved_disps[2*n_p_i+j, i] = achieved_disp

    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments', 'Paper', 'Cache')

    id_params = (
        n, p, p_is, mc_iterations, max_t, interval, seed, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution,
        epsilon_clustering)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname, tag + '_metrics_corrected.p'), 'wb') as f:
        out = [Cs, peak_times_1, peak_heights_1, period_prevalences_1,
               peak_times_2, peak_heights_2, period_prevalences_2,
               peak_times_3, peak_heights_3, period_prevalences_3,
               achieved_clusterings, achieved_disps]

        pickle.dump(out, f)

    # two modes for visualization
    scale = 1

    fig, axes = plt.subplots(4, 1, figsize=(columwidth, 2*columwidth), dpi=1000)


    # fig.subplots_adjust(wspace = 0.5)
    # (axul, axur), (axll, axlr) = axes # upper left, upper right, lower left, lower right
    ax1, ax2, ax3, ax4 = axes # reordered to be 4x1.


    # ax1.set_ylabel('$\\alpha_q$')
    # ax3.set_ylabel('$\\alpha_t$')
    # ax2.set_ylabel('$\\beta_q$')
    # ax4.set_ylabel('$\\beta_t$')

    ax1.set_ylabel('peak ratio')
    ax3.set_ylabel('peak ratio')
    ax2.set_ylabel('overall ratio')
    ax4.set_ylabel('overall ratio')


    # ax1.set_ylabel('Scaled peak height')
    # ax3.set_ylabel('Scaled peak height')
    # ax2.set_ylabel('Scaled period prevalence')
    # ax4.set_ylabel('Scaled period prevalence')
    # ax1.set_ylabel('Scaled peak height')
    # ax3.set_ylabel('Scaled peak height')
    # ax2.set_ylabel('Scaled period prevalence')
    # ax4.set_ylabel('Scaled period prevalence')
    # ax2.set_xlabel('C(g)')
    ax4.set_xlabel('C(g)')

    # axll.set_xticks(Cs, minor=False)
    # axll.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axlr.set_xticks(Cs, minor=False)
    # axlr.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # ax1.set_title('Quarantine')
    # ax3.set_title('Tracing')


    #ax1.set_prop_cycle(color=['orange','orange','orange',],linestyle=['-','--',':'])
    #ax2.set_prop_cycle(color=['orange','orange','orange',],linestyle=['-','--',':'])
    #ax3.set_prop_cycle(color=['green','green','green',],linestyle=['-','--',':'])
    #ax4.set_prop_cycle(color=['green','green','green',],linestyle=['-','--',':'])

    oranges = plt.get_cmap('Oranges')
    greens = plt.get_cmap('Greens')
    n_colors = len(p_is)
    col_vals = np.linspace(0.35,1,n_colors)

    colors = [oranges(col_vals[0]),oranges(col_vals[1]),oranges(col_vals[2]),greens(col_vals[0]),greens(col_vals[1]),greens(col_vals[2]),]
    linestyles = ['-','--',':']

    line_artists = [None,]*2*len(p_is)

    for i in range(3):
        linestyle = linestyles[i]
        l1 = ax1.plot(Cs, peak_heights_2[:,i]/peak_heights_1[:,i],color=colors[i],linestyle = linestyle, zorder=1)
        estimateQuotientCI(ax1,Cs,peak_heights_2[:,i],peak_heights_sd_2[:,i],peak_heights_1[:,i],peak_heights_sd_1[:,i],color=colors[i], mccount = mc_iterations, p=95)
        l3 = ax3.plot(Cs, peak_heights_3[:,i]/peak_heights_1[:,i],color=colors[3+i],linestyle = linestyle,zorder=1)
        estimateQuotientCI(ax3, Cs, peak_heights_3[:, i], peak_heights_sd_3[:, i], peak_heights_1[:, i],
                           peak_heights_sd_1[:, i], color=colors[3+i], mccount = mc_iterations, p=95)
        l2 = ax2.plot(Cs, period_prevalences_2[:,i]/period_prevalences_1[:,i],color=colors[i],linestyle = linestyle,zorder=1)
        estimateQuotientCI(ax2, Cs, period_prevalences_2[:, i], period_prevalences_sd_2[:, i], period_prevalences_1[:, i],
                           period_prevalences_sd_1[:, i], color=colors[i], mccount = mc_iterations,p=95)
        l4 = ax4.plot(Cs, period_prevalences_3[:,i]/period_prevalences_1[:,i],color=colors[3+i],linestyle = linestyle,zorder=1)
        estimateQuotientCI(ax4, Cs, period_prevalences_3[:, i], period_prevalences_sd_3[:, i],
                           period_prevalences_1[:, i],
                           period_prevalences_sd_1[:, i], color=colors[3+i], mccount = mc_iterations, p=95)
        line_artists[i] = l1[0]
        line_artists[len(p_is) + i] = l3[0]



    labels1 = list(['quarantine: $p_i$=' + str(val) for val in p_is])
    labels2 = list(['tracing: $p_i$=' + str(val) for val in p_is])
    # line_labels = [None,]*2*len(p_is)

    # line_labels[::2] = labels1
    # line_labels[1::2] = labels2
    line_labels = labels1 + labels2

    fig.legend(handles = line_artists,     # The line objects
               labels=line_labels,   # The labels for each line
               loc="center",   # Position of legend
               bbox_to_anchor = (0.5,-0.1),
               borderaxespad=0.1,    # Small spacing around legend box
               )

    plt.subplots_adjust(bottom=0.01)

    # ax1.legend(['$p_i$=' + str(val) for val in p_is],loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3) # looks bad
    # ax2.legend(['$p_i$=' + str(val) for val in p_is],bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3, mode="expand") # looks worse
    # ax3.legend(['$p_i$=' + str(val) for val in p_is], loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=True, ncol=3)


    # # left upper axis for dispersion values
    # ax_upper_axis = axul.twiny()
    # ax_upper_axis.set_xlim(axul.get_xlim())
    # ax_upper_axis.set_xticks(Cs)
    # ax_upper_axis.set_xticklabels(["{:.2f}".format(a) for a in achieved_disps.mean(axis=0)])
    # ax_upper_axis.set_xlabel('D(g)')
    #
    # # right upper axis for dispersion values
    # ax_upper_axis = axur.twiny()
    # ax_upper_axis.set_xlim(axul.get_xlim())
    # ax_upper_axis.set_xticks(Cs)
    # ax_upper_axis.set_xticklabels(["{:.2f}".format(a) for a in achieved_disps.mean(axis=0)])
    # ax_upper_axis.set_xlabel('D(g)')


    # ax2.plot(Cs, peak_heights_2, 'C1')
    # ax2.plot(Cs, peak_heights_3, 'C2')
    # ax2.set_ylabel('Scaled peak height')
    #
    # ax2.plot(Cs, peak_heights_2, 'C1')
    # ax2.plot(Cs, peak_heights_3, 'C2')
    # ax2.set_ylabel('Scaled peak height')
    #
    # ax3.plot(Cs, period_prevalences_2, 'C1')
    # ax3.plot(Cs, period_prevalences_3, 'C2')
    # ax3.set_ylabel('Scaled period prevalence')
    # ax3.set_xlabel('C(g)')
    # # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    # ax3.set_xticks(Cs, minor=False)
    # ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #
    #
    # ax_upper_axis = ax2.twiny()
    #
    # ax_upper_axis.set_xlim(ax3.get_xlim())
    # ax_upper_axis.set_xticks(Cs)
    # ax_upper_axis.set_xticklabels(["{:.3f}".format(a) for a in achieved_disps.mean(axis=0)])
    # # ax_upper_axis.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax_upper_axis.set_xlabel('D(g)')
    #
    # # plt.xticks([interval[0],interval[1]])
    # ax3.legend(['Quarantine', 'Tracing'])
    plt.tight_layout()

    parent = os.path.dirname(path)
    fig.savefig(os.path.join(parent, 'Pics', 'Cvaried_n{}_C{}_comp_corrected'.format(
        n, str(interval[0]) + 'to' + str(interval[1])) + '.pdf'), bbox_inches='tight', pad_inches=0)
    return out


def vary_C_comp_epcurves(res, n, p, p_i, mc_iterations, max_t, interval, seed=0, force_recompute=False,
                         path=None):
    # measure effect of clustering coeff on tracing effectiveness. Here we scale according to the vanilla outcome

    # res parameter defines how many points on [0,1] are used
    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    # the following two variables save the actual values that were achieved by the heuristic.
    # In theory, these should be approximately the same in each net
    achieved_clusterings = np.zeros((3, res))
    achieved_disps = np.zeros((3, res))

    # set up the plots
    # fig, axes = plt.subplots(1, 4, figsize=(8*scale, 4*scale),gridspec_kw={'width_ratios': [5,5,5,0.3]}, dpi=1000)


    fig = plt.figure(figsize=(columwidth, columwidth))

    rows = 3
    columns = 2

    grid = fig.add_gridspec(rows, columns, wspace = .25, hspace = .35, width_ratios = [10,0.4])

    # plt.subplot(grid[0, :])
    # plt.annotate('sub1', xy = (0.5, -0.5), va = 'center', ha = 'center',  weight='bold', fontsize = 15)
    # plt.plot(x, y)
    #
    # plt.subplot(grid[1, 0])
    # plt.annotate('sub2', xy = (0.5, -0.5), va = 'center', ha = 'center',  weight='bold', fontsize = 15)
    # plt.plot(x, y)
    #
    # plt.subplot(grid[1, 1])
    # plt.annotate('sub3', xy = (0.5, -0.5), va = 'center', ha = 'center',  weight='bold', fontsize = 15)
    # plt.plot(x, y)
    #
    # plt.subplot(grid[1, 2])
    # plt.annotate('sub4', xy = (0.5, -0.5), va = 'center', ha = 'center',  weight='bold', fontsize = 15)
    # plt.plot(x, y)
    # plt.show()
    ax1 = fig.add_subplot(grid[0,0])
    ax2 = fig.add_subplot(grid[1,0])
    ax3 = fig.add_subplot(grid[2,0])

    cbar_ax = fig.add_subplot(grid[:,1])

    # ax1, ax2, ax3, cbar_ax = axes

    ax1.set_ylabel('Infected')
    ax2.set_ylabel('Infected')
    ax3.set_ylabel('Infected')
    ax1.set_xlabel('t')
    ax2.set_xlabel('t')
    ax3.set_xlabel('t')


    # cbar_ax.axis('off')

    norm = matplotlib.colors.Normalize(vmin=Cs[0], vmax=Cs[-1], clip=False)
    cmap = plt.cm.jet
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar_ax.set_ylabel('C(g)')

    # set up colorcycles
    # color = plt.cm.viridis(Cs)
    # norm = mpl.colors.Normalize(vmin=Cs[0], vmax=Cs[-1])
    # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color), ax=ax1)
    # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color), ax=ax2)
    # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color), ax=ax3)
    # color.cycle_cmap(res)
    # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    # ax3.legend(list(['C = '+ str(C) for C in Cs]))

    # vanilla
    peak_times_1 = np.ndarray(res)
    peak_heights_1 = np.ndarray(res)
    period_prevalences_1 = np.ndarray(res)
    unsuccessful_flags_1 = []
    for i, C in tqdm(enumerate(Cs),desc='Vanilla',total=res):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i, mode='vanilla',
                              force_recompute=force_recompute,
                              path=path, clustering=C)
        peak_times_1[i] = t_peak
        peak_heights_1[i] = mean_peak
        period_prevalences_1[i] = mean_prevalence
        achieved_clusterings[0, i] = achieved_clustering
        achieved_disps[0, i] = achieved_disp

        # epidemiological curve
        ax1.plot(mean_counts[2, :],color=cmap(norm(C)))

    # quarantine
    peak_times_2 = np.ndarray(res)
    peak_heights_2 = np.ndarray(res)
    period_prevalences_2 = np.ndarray(res)
    unsuccessful_flags_2 = []
    for i, C in tqdm(enumerate(Cs),desc='Quarantine',total=res):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, seed=seed + i + res, mode='quarantine',
                              force_recompute=force_recompute,
                              path=path, clustering=C)
        peak_times_2[i] = t_peak
        peak_heights_2[i] = mean_peak / peak_heights_1[i]
        period_prevalences_2[i] = mean_prevalence / period_prevalences_1[i]
        achieved_clusterings[1, i] = achieved_clustering
        achieved_disps[1, i] = achieved_disp

        # epidemiological curve
        ax2.plot(mean_counts[2, :],color=cmap(norm(C)))

    # tracing
    peak_times_3 = np.ndarray(res)
    peak_heights_3 = np.ndarray(res)
    period_prevalences_3 = np.ndarray(res)
    unsuccessful_flags_3 = []
    for i, C in tqdm(enumerate(Cs), desc='Tracing', total=res):
        net, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , achieved_clustering, achieved_disp = \
            simple_experiment(n, p, p_i, 2*mc_iterations, max_t, seed=seed + i + 2 * res, mode='tracing',
                              force_recompute=force_recompute,
                              path=path, clustering=C)
        peak_times_3[i] = t_peak
        peak_heights_3[i] = mean_peak / peak_heights_1[i]
        period_prevalences_3[i] = mean_prevalence / period_prevalences_1[i]
        achieved_clusterings[2, i] = achieved_clustering
        achieved_disps[2, i] = achieved_disp

        # epidemiological curve
        ax3.plot(mean_counts[2, :],color=cmap(norm(C)))

    parent = os.path.dirname(path)
    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent, 'Experiments', 'Paper', 'Cache')

    id_params = (
        n, p, p_i, mc_iterations, max_t, interval, seed, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution,
        epsilon_clustering)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname, tag + '_metrics_corrected.p'), 'wb') as f:
        out = [Cs, unsuccessful_flags_1, peak_times_1, peak_heights_1, period_prevalences_1,
               Cs, unsuccessful_flags_2, peak_times_2, peak_heights_2, period_prevalences_2,
               Cs, unsuccessful_flags_3, peak_times_3, peak_heights_3, period_prevalences_3,
               achieved_clusterings, achieved_disps]

        pickle.dump(out, f)

    # plt.tight_layout()
    fig.align_ylabels()


    fig.savefig(os.path.join(dirname_parent,'Experiments','Paper', 'Pics', 'Cvaried_n{}_C{}_comp_epcurves'.format(
        n, str(interval[0]) + 'to' + str(interval[1])) + '.pdf'), bbox_inches='tight')

    return out


if __name__ == '__main__':
    res = 20
    n = 500
    p = 0.1
    p_i = 0.5
    mc_iterations = 50
    max_t = 200
    path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\vary_params'

    vary_p(res=res, n=n, p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, path=path)
    vary_p(res=res, n=n, p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, mode='quarantine', force_recompute=False,
           path=path)
    vary_p(res=res, n=n, p_i=p_i, mc_iterations=mc_iterations, max_t=max_t, mode='tracing', force_recompute=False,
           path=path)

    vary_p_i(res=res, n=n, p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, path=path)
    vary_p_i(res=res, n=n, p=p, mc_iterations=mc_iterations, max_t=max_t, mode='quarantine', force_recompute=False,
             path=path)
    vary_p_i(res=res, n=n, p=p, mc_iterations=mc_iterations, max_t=max_t, mode='tracing', force_recompute=False,
             path=path)

    # vary_p(res=3,n=100,p_i=0.5, mc_iterations=1, max_t=20 path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\vary_params')
