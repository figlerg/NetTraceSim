# want to have a SEIR model that reflects a population as a network
# nodes- people
# edges- contact between people ("friends")
# 4 possible states for nodes (SEIR):
    # 0 susceptible
    # 1 exposed
    # 2 infectious
    # 3 recovered

# let's start with simple time steps and agents (might move on to events?)

import numpy as np
import networkx as nx


# parameters: number of people, initial nr of infected, random network parameters (p or M for generating network structure)
    # infection parameters (risk for exposed, time to recovered, incubation time etc)
    # time dynamics? what about time step length etc?

# MAYBE generate network isolated from actual model to easily switch between different networks


# what it should do:
    # at each time step, each infected agent can infected connected individuals
        # (keep track via matrix, rows for individuals and columns for time steps)
    # print picture at each time step (animation)
    # visualize each of the 4 states with different colour
    # can the network dynamics change? If so, how?

