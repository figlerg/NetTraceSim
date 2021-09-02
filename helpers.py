import heapq
import time

import networkx as nx
import numpy as np

def heap_delete(h:list, i):
    # from https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
    # this is O(logn)
    h[i] = h[-1]
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)

def disp(graph:nx.Graph):
    # returns the dispersion of a network
    # tic = time.time()
    n = len(graph.nodes)
    degrees = np.zeros(n)
    for i,node in enumerate(graph.nodes):
        deg = graph.degree[node]
        degrees[i] = deg

    sigma2 = degrees.var()
    mu = degrees.mean()
    # toc = time.time()

    # print(toc-tic)
    return sigma2/mu