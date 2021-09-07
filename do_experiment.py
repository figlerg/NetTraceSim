import os
import pickle
from net import Net
from globals import *
import matplotlib.pyplot as plt
import matplotlib
import hashlib


# font = {#'family' : 'normal',
#         # 'weight' : 'bold',
#         'size'   : 22}
#
# matplotlib.rc('font', **font)


# pickling disabled for now, uncomment plot lines for that
def simple_experiment(n, p, p_i, mc_iterations, max_t, mode=None, force_recompute=False, path=None,
                      clustering: float = None, dispersion=None):
    # this creates the net, runs monte carlo on it and saves the resulting timeseries plot, as well as pickles for net and counts
    # NOTE: SEED IS CONSTANT HERE!

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

    # tag = "n{0}_p{1}_mc{2}".format(n, str(p), mc_iterations)
    # if mode:
    #     tag += '_' + mode
    # if clustering:
    #     tag += '_clustering{}'.format(clustering)

    # the cache is now tagged with a hash from all important parameters instead of the above.
    # Any change to the model parameters will certainly trigger a recompute now
    id_params = (n, p, p_i, mc_iterations, max_t, mode, clustering,dispersion, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution, chosen_epsilon)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())


    # disables loading pickled results
    if force_recompute:
        # if false, it looks at saved experiments and reuses those
        net = Net(n=n, p=p, p_i=p_i, max_t=max_t, seed=123, clustering_target=clustering,dispersion_target=dispersion)
        counts, sd = net.monte_carlo(mc_iterations, mode=mode)
        with open(os.path.join(dirname, tag + '_net.p'), 'wb') as f:
            pickle.dump(net, f)
        with open(os.path.join(dirname, tag + '_counts.p'), 'wb') as f:
            pickle.dump((counts,sd), f)

        # net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    else:
        try:
            with open(os.path.join(dirname, tag + "_counts.p"), 'rb') as f:
                counts,sd = pickle.load(f)
            with open(os.path.join(dirname, tag + "_net.p"), 'rb') as f:
                net = pickle.load(f)
            # with open(os.path.join(dirname,tag+"_vis.png"), 'rb') as f:
            #     pass

            print('Experiment results have been loaded from history.')

        except FileNotFoundError:
            net = Net(n=n, p=p, p_i=p_i, max_t=max_t, seed=123, clustering_target=clustering, dispersion_target = dispersion)

            counts, sd = net.monte_carlo(mc_iterations, mode=mode)
            with open(os.path.join(dirname, tag + '_net.p'), 'wb') as f:
                pickle.dump(net, f)
            with open(os.path.join(dirname, tag + '_counts.p'), 'wb') as f:
                pickle.dump((counts,sd), f)

            # net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    exposed = counts[EXP_STATE, :]
    infected = counts[INF_STATE, :]
    ep_curve = exposed + infected

    # compute when the peak happens and what the ratio of infected is then
    t_peak = np.argmax(ep_curve, axis=0)
    # if net.final_dispersion%10 >9:
    #     net.plot_timeseries(counts)
    #     plt.show()

    peak_height = ep_curve[t_peak] / n

    # compute the ratio of all exposed people at end of sim to the number of indiv.
    # (also check heuristically whether an equilibrium has been reached

    recovered = counts[REC_STATE, :]
    virus_contacts = ep_curve + recovered

    sensitivity = max(1, n / 100)  # increasing divisor makes this more sensitive
    equilib_flag = abs(
        virus_contacts[-1] - virus_contacts[-2]) < sensitivity  # just a heuristic, see whether roc is low

    period_prevalence = virus_contacts[-1] / n

    return (net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence)


def vary_p(res, n, p_i, mc_iterations, max_t, interval=(0, 1), mode=None, force_recompute=False, path=None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, period_prevalence

    # res parameter defines how many points on [0,1] are used
    res = res - 1 # makes more sense to me because of linspace
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    period_prevalences = np.ndarray(res)

    ps = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    for i, p in enumerate(ps):
        net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, mode, force_recompute=force_recompute, path=path)

        peak_times[i] = t_peak
        peak_heights[i] = peak_height
        period_prevalences[i] = period_prevalence

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16 * 1.5, 9 * 1.5))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    if mode:
        ax1.set_title(mode)
    else:
        ax1.set_title('vanilla')

    ax1.plot(ps, peak_times)
    # ax1.set_xlabel('p')
    ax1.set_ylabel('Peak time')

    ax2.plot(ps, peak_heights)
    # ax2.set_xlabel('p')
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(ps, period_prevalences)
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('p')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(ps[1:-2],minor=True)
    ax3.set_xticks([interval[0],interval[1]])

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # both major and minor ticks are affected
        # bottom=False,      # ticks along the bottom edge are off
        # top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])


    if mode:
        fig.savefig(os.path.join(path, 'pvaried_n{}_p{}_{}'.format(
            n,str(interval[0])+'to'+str(interval[1]), mode) + '.png'))
    else:
        fig.savefig(os.path.join(path, 'pvaried_n{}_p{}'.format(
            n, str(interval[0])+'to'+str(interval[1])) + '.png'))


def vary_p_plot_cache(res, n, p_i, mc_iterations, max_t, interval=(0, 1), force_recompute=False, path=None):
    # utility function that loads all the pickles (or runs them first) and plots the three scenarios
    # is a modified copy of vary_p !

    # res parameter defines how many points on [0,1] are used
    res = res - 1 # makes more sense to me because of linspace
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    period_prevalences = np.ndarray(res)
    peak_times_q = np.ndarray(res)
    peak_heights_q = np.ndarray(res)
    period_prevalences_q = np.ndarray(res)
    peak_times_t = np.ndarray(res)
    peak_heights_t = np.ndarray(res)
    period_prevalences_t = np.ndarray(res)

    ps = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    # all 3 modes
    for i, p in enumerate(ps):
        net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, mode=None, force_recompute=force_recompute, path=path)

        peak_times[i] = t_peak
        peak_heights[i] = peak_height
        period_prevalences[i] = period_prevalence

    for i, p in enumerate(ps):
        net_q, counts_q, sd_q, t_peak_q, peak_height_q, equilib_flag_q, period_prevalence_q = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, mode='quarantine', force_recompute=force_recompute,
                              path=path)

        peak_times_q[i] = t_peak_q
        peak_heights_q[i] = peak_height_q
        period_prevalences_q[i] = period_prevalence_q

    for i, p in enumerate(ps):
        net_t, counts_t, sd_t, t_peak_t, peak_height_t, equilib_flag_t, period_prevalence_t = \
            simple_experiment(n, p, p_i, mc_iterations, max_t, mode='tracing', force_recompute=force_recompute,
                              path=path)

        peak_times_t[i] = t_peak_t
        peak_heights_t[i] = peak_height_t
        period_prevalences_t[i] = period_prevalence_t

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14/16*9))
    ax1, ax2, ax3 = axes


    ax1.plot(ps, peak_times, ps, peak_times_q, ps, peak_times_t)
    ax1.set_ylabel('Peak time')

    ax2.plot(ps, peak_heights,ps, peak_heights_q, ps, peak_heights_t)
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(ps, period_prevalences, ps, period_prevalences_q, ps, period_prevalences_t)
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('p')
    ax3.set_xticks(ps[1:-2],minor=True)
    ax3.set_xticks([interval[0],interval[1]])

    plt.legend(['Vanilla', 'Quarantine', 'Tracing'])

    plt.tick_params(
        axis='x',
        which='minor',
        # bottom=False,      # ticks along the bottom edge are off
        # top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    parent = os.path.dirname(path)
    fig.savefig(os.path.join(parent,'Pics', 'pvaried_n{}_mc{}_{}'.format(n, mc_iterations, 'comp') + '.png'),bbox_inches = 'tight')

    # if mode:
    #     fig.savefig(os.path.join(path, 'pvaried_n{}_pi{}_{}'.format(n, p_i, mode) + '.png'))
    # else:
    #     fig.savefig(os.path.join(path, 'pvaried_n{}_pi{}'.format(n, p_i) + '.png'))








# this feels pretty uninteresting:
def vary_p_i(res, n, p, mc_iterations, max_t, mode=None, force_recompute=False, path=None):
    # here I want to systematically check what varying the edge probability does. Should return something like a 1d heatmap?
    # return value should use one of the values t_peak, peak_height, equilib_flag, period_prevalence

    # res parameter defines how many points on [0,1] are used
    res = res - 1  # silly, but for me it makes more sense to create n values for res = n (linspace creates n+1 with endpoint)
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    # flags = np.ndarray(res)
    period_prevalences = np.ndarray(res)

    p_is = np.linspace(0, 1, endpoint=True, num=res)

    for i, p_inf in enumerate(p_is):
        net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
            simple_experiment(n, p, p_inf, mc_iterations, max_t, mode, force_recompute=force_recompute, path=path)
        # TODO seed inside simple_experiment is constant, think about whether that's ok!

        peak_times[i] = t_peak
        peak_heights[i] = peak_height
        period_prevalences[i] = period_prevalence

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


def vary_C(res, n, p, p_i, mc_iterations, max_t, interval=None, mode=None, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5*p,10*p)

    # res parameter defines how many points on [0,1] are used
    res = res - 1 # makes more sense to me because of linspace
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    period_prevalences = np.ndarray(res)

    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    unsuccessful_flag = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode, force_recompute=force_recompute, path=path,
                                  clustering=C)
            peak_times[i] = t_peak
            peak_heights[i] = peak_height
            period_prevalences[i] = period_prevalence


            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flag.append(i)
            peak_times[i] = np.nan
            peak_heights[i] = np.nan
            period_prevalences[i] = np.nan



    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent,'Experiments','Paper', 'Cache')

    id_params = (n, p, p_i, mc_iterations, max_t, mode, interval, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution, epsilon_disp,'disp')
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname,tag+'_metrics.p'),'wb') as f:
        out = [Cs, unsuccessful_flag,peak_times, peak_heights,period_prevalences]

        pickle.dump(out, f)





    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14/16*9))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    colordict = {'vanilla':'C0','quarantine':'C1','tracing':'C2'}

    if mode:
        ax1.set_title(mode)
    else:
        ax1.set_title('Vanilla')

    ax1.plot(Cs, peak_times,colordict[mode])
    ax1.set_ylabel('Peak time')

    ax2.plot(Cs, peak_heights,colordict[mode])
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(Cs, period_prevalences,colordict[mode])
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Cs[1:-1],minor=True)
    ax3.set_xticks([interval[0],interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    if mode:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent,'Pics', 'Cvaried_n{}_p{}_{}'.format(
            n,str(interval[0])+'to'+str(interval[1]), mode) + '.png'),bbox_inches = 'tight')
    else:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent,'Pics', 'Cvaried_n{}_C{}'.format(
            n,str(interval[0])+'to'+str(interval[1])) + '.png'),bbox_inches = 'tight')

    return out # Cs, unsuccessful_flags, times, peaks, period_prev

def vary_disp(res, n, p, p_i, mc_iterations, max_t, interval=None, mode=None, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5*p,10*p)

    # res parameter defines how many points on [0,1] are used
    res = res - 1 # makes more sense to me because of linspace
    peak_times = np.ndarray(res)
    peak_heights = np.ndarray(res)
    period_prevalences = np.ndarray(res)

    Ds = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    unsuccessful_flag = []
    for i, D in enumerate(Ds):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode, force_recompute=force_recompute, path=path,
                                  dispersion=D)
            peak_times[i] = t_peak
            peak_heights[i] = peak_height
            period_prevalences[i] = period_prevalence

            print('last_disp{}, target_disp{}'.format(net.final_dispersion, D))


            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Dispersion target not reached')

            unsuccessful_flag.append(i)
            peak_times[i] = np.nan
            peak_heights[i] = np.nan
            period_prevalences[i] = np.nan



    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent,'Experiments','Paper', 'Cache')

    id_params = (n, p, p_i, mc_iterations, max_t, mode, interval, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution, epsilon)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname,tag+'_metrics.p'),'wb') as f:
        out = [Ds, unsuccessful_flag,peak_times, peak_heights,period_prevalences]

        pickle.dump(out, f)





    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14/16*9))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes

    colordict = {'vanilla':'C0','quarantine':'C1','tracing':'C2'}

    if mode:
        ax1.set_title(mode)
    else:
        ax1.set_title('Vanilla')

    ax1.plot(Ds, peak_times,colordict[mode])
    ax1.set_ylabel('Peak time')

    ax2.plot(Ds, peak_heights,colordict[mode])
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(Ds, period_prevalences,colordict[mode])
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Ds[1:-1],minor=True)
    ax3.set_xticks([interval[0],interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])

    if mode:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent,'Pics', 'dispvaried_n{}_p{}_{}'.format(
            n,str(interval[0])+'to'+str(interval[1]), mode) + '.png'),bbox_inches = 'tight')
    else:
        parent = os.path.dirname(path)
        fig.savefig(os.path.join(parent,'Pics', 'dispvaried_n{}_C{}'.format(
            n,str(interval[0])+'to'+str(interval[1])) + '.png'),bbox_inches = 'tight')

    return out # Cs, unsuccessful_flags, times, peaks, period_prev


def vary_C_comp(res, n, p, p_i, mc_iterations, max_t, interval=None, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5*p,10*p)

    # res parameter defines how many points on [0,1] are used
    res = res - 1 # makes more sense to me because of linspace
    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    peak_times_1 = np.ndarray(res)
    peak_heights_1 = np.ndarray(res)
    period_prevalences_1 = np.ndarray(res)
    unsuccessful_flags_1 = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode='vanilla', force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_1[i] = t_peak
            peak_heights_1[i] = peak_height
            period_prevalences_1[i] = period_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_1.append(i)
            peak_times_1[i] = np.nan
            peak_heights_1[i] = np.nan
            period_prevalences_1[i] = np.nan

    peak_times_2 = np.ndarray(res)
    peak_heights_2 = np.ndarray(res)
    period_prevalences_2 = np.ndarray(res)
    unsuccessful_flags_2 = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode='quarantine', force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_2[i] = t_peak
            peak_heights_2[i] = peak_height
            period_prevalences_2[i] = period_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_2.append(i)
            peak_times_2[i] = np.nan
            peak_heights_2[i] = np.nan
            period_prevalences_2[i] = np.nan


    peak_times_3 = np.ndarray(res)
    peak_heights_3 = np.ndarray(res)
    period_prevalences_3 = np.ndarray(res)
    unsuccessful_flags_3 = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode='tracing', force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_3[i] = t_peak
            peak_heights_3[i] = peak_height
            period_prevalences_3[i] = period_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_3.append(i)
            peak_times_3[i] = np.nan
            peak_heights_3[i] = np.nan
            period_prevalences_3[i] = np.nan





    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent,'Experiments','Paper', 'Cache')

    id_params = (n, p, p_i, mc_iterations, max_t, interval, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution, epsilon_clustering)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname,tag+'_metrics.p'),'wb') as f:
        out = [Cs, unsuccessful_flags_1,peak_times_1, peak_heights_1,period_prevalences_1,
               Cs, unsuccessful_flags_2,peak_times_2, peak_heights_2,period_prevalences_2,
               Cs, unsuccessful_flags_3,peak_times_3, peak_heights_3,period_prevalences_3]

        pickle.dump(out, f)





    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 14/16*9))

    # fig.subplots_adjust(wspace = 0.5)
    ax1, ax2, ax3 = axes


    ax1.plot(Cs, peak_times_1,Cs, peak_times_2,Cs, peak_times_3)
    ax1.set_ylabel('Peak time')

    ax2.plot(Cs, peak_heights_1,Cs, peak_heights_2,Cs, peak_heights_3)
    ax2.set_ylabel('Peak prevalence')

    ax3.plot(Cs, period_prevalences_1,Cs, period_prevalences_2,Cs, period_prevalences_3)
    ax3.set_ylabel('Fraction of affected')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Cs[1:-1],minor=True)
    ax3.set_xticks([interval[0],interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])
    plt.legend(['Vanilla', 'Quarantine', 'Tracing'])


    parent = os.path.dirname(path)
    fig.savefig(os.path.join(parent,'Pics', 'Cvaried_n{}_C{}_comp'.format(
        n,str(interval[0])+'to'+str(interval[1])) + '.png'),bbox_inches = 'tight')

    return out # Cs, unsuccessful_flags, times, peaks, period_prev

def vary_C_comp_corrected(res, n, p, p_i, mc_iterations, max_t, interval=None, force_recompute=False, path=None):
    # measure effect of clustering coeff on tracing effectiveness. Here we scale according to the vanilla outcome

    if not interval:
        # THEORY: the average clustering coeff of erdos renyi networks is p!
        # so I test around that to see what changed
        interval = (0.5*p,10*p)

    # res parameter defines how many points on [0,1] are used
    res = res - 1 # makes more sense to me because of linspace
    Cs = np.linspace(interval[0], interval[1], endpoint=True, num=res)

    # vanilla
    peak_times_1 = np.ndarray(res)
    peak_heights_1 = np.ndarray(res)
    period_prevalences_1 = np.ndarray(res)
    unsuccessful_flags_1 = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode='vanilla', force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_1[i] = t_peak
            peak_heights_1[i] = peak_height
            period_prevalences_1[i] = period_prevalence

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_1.append(i)
            peak_times_1[i] = np.nan
            peak_heights_1[i] = np.nan
            period_prevalences_1[i] = np.nan

    # quarantine
    peak_times_2 = np.ndarray(res)
    peak_heights_2 = np.ndarray(res)
    period_prevalences_2 = np.ndarray(res)
    unsuccessful_flags_2 = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode='quarantine', force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_2[i] = t_peak
            peak_heights_2[i] = peak_height / peak_heights_1[i]
            period_prevalences_2[i] = period_prevalence / period_prevalences_1[i]

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_2.append(i)
            peak_times_2[i] = np.nan
            peak_heights_2[i] = np.nan
            period_prevalences_2[i] = np.nan

    # tracing
    peak_times_3 = np.ndarray(res)
    peak_heights_3 = np.ndarray(res)
    period_prevalences_3 = np.ndarray(res)
    unsuccessful_flags_3 = []
    for i, C in enumerate(Cs):
        try:
            net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence = \
                simple_experiment(n, p, p_i, mc_iterations, max_t, mode='tracing', force_recompute=force_recompute,
                                  path=path, clustering=C)
            peak_times_3[i] = t_peak
            peak_heights_3[i] = peak_height / peak_heights_1[i]
            period_prevalences_3[i] = period_prevalence / period_prevalences_1[i]

            # Cs[i] = net.final_cluster_coeff # in the end I want to plot the actual coeff, not the target
            # should specify this in the paper
        except AssertionError:
            print('Clustering target not reached')

            unsuccessful_flags_3.append(i)
            peak_times_3[i] = np.nan
            peak_heights_3[i] = np.nan
            period_prevalences_3[i] = np.nan





    dirname_parent = os.path.dirname(__file__)
    dirname = os.path.join(dirname_parent,'Experiments','Paper', 'Cache')

    id_params = (n, p, p_i, mc_iterations, max_t, interval, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution, epsilon_clustering)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    with open(os.path.join(dirname,tag+'_metrics_corrected.p'),'wb') as f:
        out = [Cs, unsuccessful_flags_1,peak_times_1, peak_heights_1,period_prevalences_1,
               Cs, unsuccessful_flags_2,peak_times_2, peak_heights_2,period_prevalences_2,
               Cs, unsuccessful_flags_3,peak_times_3, peak_heights_3,period_prevalences_3]

        pickle.dump(out, f)





    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 14/16*9))

    # fig.subplots_adjust(wspace = 0.5)
    ax2, ax3 = axes


    # ax1.plot(Cs, peak_times_1,Cs, peak_times_2,Cs, peak_times_3)
    # ax1.set_ylabel('Peak time')

    ax2.plot(Cs, peak_heights_2,Cs, peak_heights_3)
    ax2.set_ylabel('Scaled peak height')

    ax3.plot(Cs, period_prevalences_2,Cs, period_prevalences_3)
    ax3.set_ylabel('Scaled period prevalence')
    ax3.set_xlabel('C(g)')
    # labels = [interval[0],] + list(['' for i in range(len(ps)-2)]) + [interval[1],]
    ax3.set_xticks(Cs[1:-1],minor=True)
    ax3.set_xticks([interval[0],interval[1]])

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='minor',      # both major and minor ticks are affected
    #     # bottom=False,      # ticks along the bottom edge are off
    #     # top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # plt.xticks([interval[0],interval[1]])
    plt.legend(['Quarantine', 'Tracing'])


    parent = os.path.dirname(path)
    fig.savefig(os.path.join(parent,'Pics', 'Cvaried_n{}_C{}_comp'.format(
        n,str(interval[0])+'to'+str(interval[1])) + '.png'),bbox_inches = 'tight')

    return out # Cs, unsuccessful_flags, times, peaks, period_prev

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
    vary_p_i(res=res, n=n, p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='quarantine',
             path=path)
    vary_p_i(res=res, n=n, p=p, mc_iterations=mc_iterations, max_t=max_t, force_recompute=False, mode='tracing',
             path=path)

    # vary_p(res=3,n=100,p_i=0.5, mc_iterations=1, max_t=20 path = r'C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\vary_params')
