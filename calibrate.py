import os
import pickle
import random

import numpy as np
import scipy.optimize
from net import Net
from globals import *
import matplotlib.pyplot as plt
import matplotlib
import hashlib
from do_experiment_parallel import *

def fit_bisection(n, p, p_i_ref, c_ref, c_range, mc_iterations, max_t, path):
    (net_ref, counts_ref, sd_ref, t_peak_ref, peak_height_ref, equilib_flag_ref, period_prevalence_ref) = simple_experiment(n, p, p_i_ref,mc_iterations,
                                                                                                            max_t,
                                                                                                            path=path,
                                                                             clustering=c_ref)
    tol = 0.01
    phs =list()
    prevs = list()
    i_range = list()
    lg = list()
    for c in c_range:
        if c==c_ref:
            (net, counts, sd, t_peak, peak_height, equilib_flag,
             period_prevalence) = simple_experiment(n, p, p_i_ref, mc_iterations,
                                                        max_t,
                                                        path=path,
                                                        clustering=c_ref)
            i_range.append(p_i_ref)
        else:
            p_low = 0.2*p_i_ref
            p_high = 2.0*p_i_ref
            while p_high - p_low >tol:
                print([p_high,p_low])
                p_mid = 0.5*(p_high + p_low)
                (net, counts, sd, t_peak, peak_height, equilib_flag, period_prevalence) = simple_experiment(n, p, p_mid,
                                                                                                            mc_iterations,
                                                                                                            max_t,
                                                                                                            path=path,
                                                                                                            clustering=c)
                if peak_height<peak_height_ref:
                    p_low = p_low + 0.5*(p_mid-p_low)  #0.5 to filter for stochasticity!
                else:
                    p_high = p_high - 0.5*(p_high-p_mid)
            i_range.append(p_mid)
        print(peak_height)
        phs.append(peak_height)
        prevs.append(period_prevalence)
        plt.plot(counts[1]+counts[2])
        lg.append(str([c,i_range[-1]]))
    plt.legend(lg)
    return i_range, phs, prevs

def calibFun(param,pp,pc,pi):
    lhs = pp*param[0]+pc*param[1]
    diff = (lhs-pi)
    return sum(diff*diff)

if __name__ == '__main__':
    random.seed(12345)
    res = 20
    n = 500
    p = 0.1

    mc_iterations = 200
    max_t = 200
    path = 'evals'

    p_i_ref = 0.5
    c_ref = 0.3
    c_range = [0.1,0.15,0.2,0.25,0.3]
    pis,phs,prevs = fit_bisection(n, p, p_i_ref, c_ref, c_range, mc_iterations, max_t, path)
    print(phs)
    print(prevs)
    plt.show()