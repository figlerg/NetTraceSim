#  for running monte carlo simulations of the simulations in net.py

from typing import List
from net import Net
import numpy as np


def monte_carlo(net: Net, n):
    # net is input
    # run sim n times, saving the output in list
    results: List[np.ndarray] = []
    for i in range(n):
        results.append(net.sim(seed=i).copy())
        # print(i)
        # net.plot_timeseries()
        net.reset()
    # compute mean

    mean = np.zeros(results[0].shape)
    for counts in results:
        mean += counts
    mean /= len(results)

    net.plot_timeseries(counts=mean)


if __name__ == '__main__':
    net = Net(n=1000, p=0.1, seed=123, max_t=100)
    # net.draw()

    monte_carlo(net, 10)
