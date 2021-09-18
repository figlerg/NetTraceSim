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
        self.max_t=max_t
        self.seed = seed
        self.clustering_target = clustering_target
        self.dispersion_target = dispersion_target
        self.mc_iterations = mc_iterations
        self.mode = mode
        self.queue = queue

    def run(self):
        with HiddenPrints():
            [mean,sd,clustering, dispersion] = Net(n=self.n, p=self.p, p_i=self.p_i, max_t=self.max_t, seed=self.seed, clustering_target=self.clustering_target,dispersion_target=self.dispersion_target).monte_carlo(self.mc_iterations, mode=self.mode)
            self.queue[self.name]=[mean,sd,clustering,dispersion]
        return

# pickling disabled for now, uncomment plot lines for that
def simple_experiment(n, p, p_i, mc_iterations, max_t, seed = 123, mode=None, force_recompute=False, path=None,
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
    id_params = (n, p, p_i, mc_iterations, max_t, seed, mode, clustering,dispersion, t_i, t_c, t_r, t_d, t_t, p_q, p_t, quarantine_time, resolution, chosen_epsilon)
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
                counts,sd = pickle.load(f)
            #with open(os.path.join(dirname, tag + "_net.p"), 'rb') as f:
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
                    MCProcess('MCProcess_' + str(i),q, n, p, p_i, max_t, seed + i, clustering, dispersion, runsPerKernel,
                              mode))
                threads[-1].start()

            countss = list()
            sds = list()
            clusterings = list()
            disps = list()
            for t in threads:
                t.join() # wait for all threads to finish
            for v in q.values():
                countss.append(v[0])
                sds.append(v[1])
                clusterings.append(v[2])
                disps.append(v[3])
            for t in threads:
                t.kill()
            counts = sum(countss) / len(countss)
            sd = sum(sds) / len(sds)  # i think this is sloppy...
            clustering = np.mean(clusterings)
            dispersion = np.mean(disps)

            #with open(os.path.join(dirname, tag + '_net.p'), 'wb') as f:
            #    pickle.dump(net, f)
            with open(os.path.join(dirname, tag + '_counts.p'), 'wb') as f:
                pickle.dump((counts,sd), f)

            # net.plot_timeseries(counts, save= os.path.join(dirname, tag+'_vis.png'))

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

    return None, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence, clustering, dispersion