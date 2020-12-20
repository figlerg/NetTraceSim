# try to use whole network as class that runs in environment

import simpy as sp
import networkx as nx
import numpy as np
# from person import Person
import matplotlib.pyplot as plt
import random

p_i = 0.1  # probability of infection at contact
t_i = 2  # incubation time
t_r = 14  # recovery time
t_c = 1  # rate of contacts



class Net(object):

    def __init__(self, env:sp.Environment, n, p, seed = 12345):

        self.n = n
        self.env = env
        self.graph = nx.gnp_random_graph(n, p, seed = seed)
        self.colormap = ['green' for i in range(n)]
        random.seed(seed)


        for id in range(n):
            # at first all are susceptible
            # print(net.nodes)
            # print(net.edges)
            self.graph.nodes[id]['state'] = 0

        self.action = self.env.process(self.infection(1))

    # events:

    def infection(self, id):
        env = self.env

        self.update_state(id,1) # exposed now
        self.colormap[id] = 'yellow'
        print('Person #{} has been exposed at time {}'.format(id, env.now))
        yield env.timeout(delay=t_i)

        # schedule infectious event
        env.process(self.infectious(id))


    def infectious(self, id):
        env = self.env
        print('Person #{} started being infectious at time {}'.format(id, env.now))
        self.update_state(id,2)
        self.colormap[id] = 'red'

        self.draw()
        plt.show()

        env.process(self.recover(id))

        try:
            yield env.timeout(delay=t_c)
            env.process(self.contact(id))
        except:
            print('Contact process stopped due to recovery.')


        # TODO schedule contact & exposed

    def contact(self, id):
        try:
            yield self.env.timeout(delay=t_c)

            friends = list(self.graph.neighbors(id))
            contacted_friend = random.choice(friends)

            # if self.graph.nodes[id]['state'] == 3:
            #     yield

            if self.graph.nodes[contacted_friend]['state'] == 0:
                u = random.uniform(0,1)

                if u < p_i:
                    self.env.process(self.infection(contacted_friend))
            # else:
            #     pass  # if in any other state than susceptible, this contact does not matter

            print('#' + str(id) + ' has had contact with #{}.'.format(contacted_friend))


            self.graph.nodes[id]['latest_contact'] = self.env.process(self.contact(id))
            # this stores a pointer to the latest contact process of this id
            # it can be used to interrupt said process should the patient recover in the meantime

            # TODO look over the order here... not sure if it makes sense to store the reference of the
            #  next contact after it happens?
        except sp.Interrupt:
            print('Contact process stopped due to recovery.')



    def recover(self, id):
        yield self.env.timeout(14)
        self.graph.nodes[id]['latest_contact'].interrupt()
        self.colormap[id] = 'grey'
        print(str(id)+' has recovered.')


    # convenience:

    def draw(self):
        pos = nx.spring_layout(self.graph, seed=100)
        # i deliberately leave the seed fixed, maybe I want same positions for networks of equal size
        nx.draw(self.graph, node_color = self.colormap, pos = pos)
        plt.show()

    def update_state(self, id, state):
        self.graph.nodes[id]['state'] = state

