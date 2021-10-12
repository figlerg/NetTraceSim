import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from helpers import HiddenPrints

# plt.style.use('seaborn-poster')
from do_experiment import simple_experiment
parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true', default=False)
args = parser.parse_args()
force_recompute = args.recompute



n=500
max_t = 125


beta = 0.3
sigma = 1.7
gamma = 0.11


def seir_f(t, y, beta, sigma, gamma):
    s, e, i, r = y
    return np.array([-beta * i * s/n,
                     -sigma * e + beta * i * s/n,
                     -gamma * i + sigma * e,
                     gamma * i])

SEIR_0 = [500,1,0,0]

t_eval = np.arange(0, max_t,0.2)
sol = solve_ivp(seir_f, [0, max_t], SEIR_0, t_eval=t_eval,args=(beta, sigma, gamma))

fig,(ax1,ax2) = plt.subplots(1,2, figsize=(14,3),squeeze=True)
ax1.set_prop_cycle(color=['green', 'yellow', 'red', 'grey'])  # same as net colormap
ax1.plot(sol.t,sol.y[0], sol.t,sol.y[1],sol.t,sol.y[2],sol.t,sol.y[3])
ax1.set_title('ODE SEIR-Model')
# ax1.legend(['S','E','I','R'])



#CALC for my model
res = 5
p = 10/500
p_i = 0.5
mc_iterations = 100

working_dir = os.getcwd()
path = os.path.join(working_dir,'Cache')

if __name__ == '__main__':

    net1, counts1,sd1, t_peak1, peak_height1, equilib_flag1, durchseuchung1 = \
        simple_experiment(n, p, p_i, mc_iterations, max_t, mode=None, force_recompute=force_recompute, path=path)


    net1.plot_timeseries(counts1,existing_ax=ax2)

    ax2.set_title('Our SEIR-Model')

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(working_dir,'Pics','ODEcomp'))