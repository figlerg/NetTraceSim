# want to have a SEIR model that reflects a population as a network
# nodes- people
# edges- contact between people ("friends")
# 4 possible states for nodes (SEIR):
    # 0 susceptible
    # 1 exposed
    # 2 infectious
    # 3 recovered

# events are actually easier? hard to make agents time things right otherwise (recovery time etc)

import numpy as np
import networkx as nx
import simpy as sp
import matplotlib.pyplot as plt
from person import Person
from net import Net


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

n = 20
p = 0.05 # probability of friendship between each individual
init_infected = 1
time = 30 # in days

incubation_time = 2
recovery_time = 14
risk = 0.1 # probability of infection per day for two friends


# init
state_graph = np.zeros((n, time),dtype=int) # list of states of each individual for each day
# encode as:
    # 0:susceptible
    # 1:exposed
    # 2:infected
    # 3:recovered

state_graph[0:init_infected,0] = 2



net = Net(n, p)


net.sim()

# test = Person(env, id = 10, state = 1)
# test.infection(1)
# print(test)




# nx.draw(net)
# # plt.show()





