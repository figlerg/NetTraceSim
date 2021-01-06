# this might be bad practise, but I am using some global variables in order to keep function signatures minimal
# they are saved here for now

import numpy as np

p_i = 0.5  # probability of infection at contact
t_i = 2  # incubation time
t_r = 14  # recovery time
t_c = 1  # time between contacts

# NOTE: this is the same as the scale parameter in np.random.exponential! No inverse needed! Checked experimentally...

resolution = 1  # days for each animation frame, abtastrate

INFECTION = 0
INFECTIOUS = 1
CONTACT = 2
RECOVER = 3

# update the counts in the events by adding these to self.count:
susc2exp = np.asarray([-1, 1, 0, 0, 0], dtype=np.int32)
exp2inf = np.asarray([0, -1, 1, 0, 0], dtype=np.int32)
inf2rec = np.asarray([0, 0, -1, 1, 0], dtype=np.int32)
inf2no_trans = np.asarray([0, 0, -1, 0, 1], dtype=np.int32)
no_trans2rec = np.asarray([0, 0, 0, 1, -1], dtype=np.int32)
