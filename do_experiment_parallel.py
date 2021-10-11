import os
import pickle
from net import Net
from globals import *
import matplotlib.pyplot as plt
import matplotlib
import hashlib
from multiprocessing import Process, Manager
from helpers import HiddenPrints


# font = {#'family' : 'normal',
#         # 'weight' : 'bold',
#         'size'   : 22}
#
# matplotlib.rc('font', **font)

class MCProcess(Process):
    def __init__(self, name, queue, n, p, p_i, max_t, seed, clustering_target, dispersion_target, mc_iterations,
                 mode):
        super().__init__(name=name)
        self.name = name
        self.n = n
        self.p = p
        self.p_i = p_i
        self.max_t = max_t
        self.seed = seed
        self.clustering_target = clustering_target
        self.dispersion_target = dispersion_target
        self.mc_iterations = mc_iterations
        self.mode = mode
        self.queue = queue

    def run(self):
        with HiddenPrints():
            [mean_counts, meansq_counts, mean_peak, meansq_peak, mean_prevalence, meansq_prevalence, clustering, dispersion] = Net(n=self.n, p=self.p, p_i=self.p_i, max_t=self.max_t, seed=self.seed,
                                                     clustering_target=self.clustering_target,
                                                     dispersion_target=self.dispersion_target).monte_carlo(
                self.mc_iterations, mode=self.mode)
			
            self.queue[self.name] = [mean_counts, meansq_counts, mean_peak, meansq_peak, mean_prevalence, meansq_prevalence, clustering, dispersion]
        return


# pickling disabled for now, uncomment plot lines for that
def simple_experiment(n, p, p_i, mc_iterations, max_t, seed=123, mode=None, force_recompute=False, path=None,
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

    # the cache is now tagged with a hash from all important parameters
    # Any change to the model parameters will certainly trigger a recompute now
    id_params = (n, p, p_i, mc_iterations, max_t, seed, mode, clustering, dispersion, t_i, t_c, t_r, t_d, t_t, p_q, p_t,
                 quarantine_time, resolution, chosen_epsilon)
    # normal hashes are salted between runs -> use something that is persistent
    tag = str(hashlib.md5(str(id_params).encode('utf8')).hexdigest())

    recompute = False
    # disables loading pickled results
    if force_recompute:
        # if false, it looks at saved experiments and reuses those
        recompute = True
    else:
        try:
            with open(os.path.join(dirname, tag + "_counts.p"), 'rb') as f:
                mean_counts, sd_counts, mean_peak, sd_peak, mean_prevalence, sd_prevalence,clustering,dispersion  = pickle.load(f)
            # with open(os.path.join(dirname, tag + "_net.p"), 'rb') as f:
            #    net = pickle.load(f)

            print('Experiment results have been loaded from history.')

        except FileNotFoundError:
            recompute = True

    if recompute:
        kernels = 7
        runsPerKernel = int(mc_iterations / 5)
        threads = list()

        q = Manager().dict()

        for i in range(kernels):
            threads.append(
                MCProcess('MCProcess_' + str(i), q, n, p, p_i, max_t, seed + i, clustering, dispersion, runsPerKernel,
                          mode))
            threads[-1].start()

        mean_counts = list()
        meansq_counts = list()
        mean_peak = list()
        meansq_peak = list()
        mean_prevalence = list()
        meansq_prevalence = list()
        clusterings = list()
        disps = list()

        for t in threads:
            t.join()  # wait for all threads to finish
        for v in q.values():
            mean_counts.append(v[0])
            meansq_counts.append(v[1])
            mean_peak.append(v[2])
            meansq_peak.append(v[3])
            mean_prevalence.append(v[4])
            meansq_prevalence.append(v[5])
            clusterings.append(v[6])
            disps.append(v[7])
        for t in threads:
            t.kill()
        mean_counts = sum(mean_counts) / len(mean_counts)
        meansq_counts = sum(meansq_counts) / len(meansq_counts)
        mean_peak = np.mean(mean_peak)
        meansq_peak = np.mean(meansq_peak)
        mean_prevalence = np.mean(mean_prevalence)
        meansq_prevalence = np.mean(meansq_prevalence)

        sd_counts = np.sqrt(meansq_counts-np.square(mean_counts))
        sd_peak = np.sqrt(meansq_peak-np.square(mean_peak))
        sd_prevalence = np.sqrt(meansq_prevalence-np.square(mean_prevalence))

        clustering = np.mean(clusterings)
        dispersion = np.mean(disps)

        # with open(os.path.join(dirname, tag + '_net.p'), 'wb') as f:
        #    pickle.dump(net, f)
        with open(os.path.join(dirname, tag + '_counts.p'), 'wb') as f:
            pickle.dump((mean_counts, sd_counts, mean_peak, sd_peak, mean_prevalence, sd_prevalence,clustering,dispersion), f)

        # net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

    exposed = mean_counts[EXP_STATE, :]
    infected = mean_counts[INF_STATE, :]
    ep_curve = exposed + infected
    t_peak = np.argmax(ep_curve, axis=0) # simply take time for peak from mean counts (sloppy)

    recovered = mean_counts[REC_STATE, :]
    virus_contacts = ep_curve + recovered

    sensitivity = max(1, n / 100)  # increasing divisor makes this more sensitive
    equilib_flag = abs(
        virus_contacts[-1] - virus_contacts[-2]) < sensitivity  # just a heuristic, see whether roc is low

    assert dispersion, 'These should not be None'

    return None, mean_counts, sd_counts, t_peak, mean_peak, sd_peak, mean_prevalence, sd_prevalence, equilib_flag , clustering, dispersion
