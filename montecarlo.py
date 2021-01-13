#  for running monte carlo simulations of the simulations in net.py

from typing import List
from net import Net
import numpy as np


def monte_carlo(net: Net, n, mode = None):
    # net is input
    # run sim n times, saving the output in list
    results: List[np.ndarray] = []
    for i in range(n):
        results.append(net.sim(seed=i, mode = mode).copy())
        # print(i)
        # net.plot_timeseries()
        net.reset()
    # compute mean

    mean = np.zeros(results[0].shape)
    for counts in results:
        mean += counts
    mean /= len(results)

    # net.plot_timeseries(counts=mean)

    return mean


if __name__ == '__main__':
    net1 = Net(n=1000, p=0.1, seed=123, max_t=100)
    # net.draw()
    counts1 = monte_carlo(net1, 10)

    net2 = Net(n=500, p=0.05, seed=123, max_t=100)
    counts2 = monte_carlo(net2, 10)

    net3 = Net(n=500, p=0.1, seed=123, max_t=100)
    counts3 = monte_carlo(net3, 10)

    import time

    for net, counts in [(net1, counts1),(net2, counts2),(net3, counts3)]:
        now = time.time()
        net.plot_timeseries(counts = counts, save = str(now) + '.jpg')

    # monte_carlo(net,10,'quarantine')
