# this might be bad practise, but I am using some global variables in order to keep function signatures minimal
# they are saved here for now

import numpy as np

# p_i = 0.5  # probability of infection at contact # TODO didnt want this global anymore, change everywhere
t_i = 2  # incubation time
t_r = 10  # recovery time
t_c = 1  # time between contacts

t_d = 6 # average time until infection is noticed
t_t = 3 # average time until contact is found and put into quarantine

quarantine_time = 14 # this should probably be a fixed time

# NOTE: this is the same as the scale parameter in np.random.exponential! No inverse needed! Checked experimentally...

resolution = 1  # days for each animation frame, abtastrate (right now only ints are possible)
# TODO maybe allow floats as well and use linspace instead of arange?

redo_net = 5 # every i iterations, monte carlo also changes network

clustering_epsilon = 0.01


INFECTION = 0
INFECTIOUS = 1
CONTACT = 2
RECOVER = 3
QUARANTINE = 4
END_OF_QUARANTINE = 5
TRACING = 6

SUSC_STATE = 0
EXP_STATE = 1
INF_STATE = 2
REC_STATE = 3
NO_TRANS_STATE = 4

# update the counts in the events by adding these to self.count:
susc2exp = np.asarray([-1, 1, 0, 0], dtype=np.int32)
exp2inf = np.asarray([0, -1, 1, 0], dtype=np.int32)
inf2rec = np.asarray([0, 0, -1, 1], dtype=np.int32)
inf2no_trans = np.asarray([0, 0, -1, 0], dtype=np.int32)
# no_trans2rec = np.asarray([0, 0, 0, 1, -1], dtype=np.int32)
# no_trans2inf = np.asarray([0, 0, 1, 0, -1], dtype=np.int32)
