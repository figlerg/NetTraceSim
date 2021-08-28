# network is a class that implements the full simulation environment
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import heapq
import matplotlib.animation as animation
import time

import numpy as np
from scipy.interpolate import interp1d
import math
from typing import List

from globals import *  # loading some variables and constants
from helpers import heap_delete


class Net(object):

    def __init__(self, n, p, p_i, max_t, seed, clustering_target=None,clustering_batchsize=None):

        # TODO try to decrease complexity, this seems convoluted

        print("Initializing network...")
        start = time.time()
        np.random.seed(seed)

        self.p_i = p_i  # infection prob at contact
        self.max_t = max_t  # sim time

        self.n = n  # nr of nodes
        self.p = p  # prob of connection between two nodes
        self.graph = nx.fast_gnp_random_graph(n, p, seed=seed)  # network structure
        self.colormap = ['green' for i in range(n)]  # for visualization in networks
        self.clustering_target = clustering_target  # the desired clustering coeff
        self.clustering_batchsize = clustering_batchsize

        self.event_list = []  # create event list as list and heapify for priority queue
        heapq.heapify(self.event_list)

        if p == 0:
            print('Warning: p = 0, so the graph will not be checked for connectedness.')
            self.graph = nx.fast_gnp_random_graph(n, p, seed=seed)
            self.last_seed = seed
        else:
            while not nx.is_connected(self.graph):
                # I only want connected graphs, otherwise i cannot really compare
                seed += 1
                self.graph = nx.fast_gnp_random_graph(n, p, seed=seed)
            else:
                self.last_seed = seed  # this is the seed that was used.
                # I save this so I can choose a different one when I want to create a new net in mc

        if self.clustering_target:
            self.final_cluster_coeff = self.alter_clustering_coeff(clustering_target, clustering_epsilon)

        # I don't want to deal with a whole mutable state list, so I only save the current count at regular intervals:
        self.count = np.zeros([4, 1], dtype=np.int32).flatten()  # current state
        # susceptible, exposed, infectious, recovered are the 4 rows

        # history, gets written in sim()
        self.count[0] = n
        self.counts = np.zeros([4, math.floor(max_t / resolution)], dtype=np.int32)

        self.net_states = []  # this is a list of nets at equidistant time steps
        # i use this for the animation frames

        # for comparison, even new network structures use same layout (this only writes self.pos once, at start)
        if not hasattr(self, 'pos'):
            self.pos = nx.spring_layout(self.graph, seed=seed)

        for id in range(n):
            # at first all agents are susceptible
            self.graph.nodes[id]['state'] = 0

        nx.set_edge_attributes(self.graph, False, name='blocked')
        nx.set_node_attributes(self.graph, [], name='contacts')

        # Here I set a reset point because some of the values are changed in place and I need a fresh
        # start for each monte carlo. Resetting is done via the self.reset() function
        self.init_state = {}
        for key in self.__dict__.keys():
            try:
                self.init_state[key] = self.__dict__[key].copy()
            except AttributeError:
                self.init_state[key] = self.__dict__[key]

        end = time.time()

        print("Network initialized. Time elapsed: {}s.".format(end - start))

    # events:

    def infection(self, time, id):

        self.update_state(id, EXP_STATE)  # exposed now
        self.count += susc2exp
        self.colormap[id] = 'yellow'
        # print('Person #{} has been exposed at time {}'.format(id, time))

        # there is a possibility that one individual gets several infection events scheduled to by different people
        # for this i created a mode for the canceling edge that cancels all scheduled events:
        self.cancel_event(id, INFECTION, all=True)

        # schedule infectious event
        t_i_random = np.random.exponential(scale=t_i, size=1)[0]
        heapq.heappush(self.event_list, (time + t_i_random, INFECTIOUS, id))

    def infectious(self, time, id, mode):
        # print('Person #{} started being infectious at time {}'.format(id, time))
        self.update_state(id, INF_STATE)
        self.count += exp2inf
        self.colormap[id] = 'red'

        t_c_random = np.random.exponential(scale=t_c, size=1)[0]
        t_r_random = np.random.exponential(scale=t_r, size=1)[0]

        heapq.heappush(self.event_list, (time + t_c_random, CONTACT, id))
        heapq.heappush(self.event_list, (time + t_r_random, RECOVER, id))

        if mode == 'quarantine' or mode == 'tracing':
            t_q_random = np.random.exponential(scale=t_d, size=1)[0]
            heapq.heappush(self.event_list, (time + t_q_random, QUARANTINE, id))

            if mode == 'tracing':
                heapq.heappush(self.event_list, (time + t_q_random, TRACING, id))
                # I will simply do these two at the same time (when the infection is detected)
                # the tracing event adds a little bit of time for the process of finding and alerting contacts

    def contact(self, time, id):

        # friends = list(self.graph.neighbors(id))
        # connections = list(self.graph.edges)
        friends = list((friend for friend in self.graph.neighbors(id)
                        if self.graph.edges[id, friend]['blocked'] == False))
        # can only use edges that aren't blocked due to quarantine

        if friends:
            contacted_friend_idx = np.random.choice(len(friends), 1)[0]
            contacted_friend = friends[contacted_friend_idx]
            self.graph.nodes[id]['contacts'].append(contacted_friend)
        else:
            t_c_random = np.random.exponential(scale=t_c, size=1)[0]
            next_contact = (time + t_c_random, CONTACT, id)
            heapq.heappush(self.event_list, next_contact)
            return

        if self.graph.nodes[contacted_friend]['state'] == SUSC_STATE:

            # print('#' + str(id) + ' has had contact with #{}.'.format(contacted_friend))
            u = np.random.uniform()

            if u < self.p_i:
                heapq.heappush(self.event_list, (time, INFECTION, contacted_friend))
        else:
            pass  # if in any other state than susceptible, this contact does not matter

        if self.graph.nodes[id]['state'] == INF_STATE:

            t_c_random = np.random.exponential(scale=t_c, size=1)[0]
            next_contact = (time + t_c_random, CONTACT, id)

            # if person is not infectious anymore, no need to schedule this
            heapq.heappush(self.event_list, next_contact)
        else:
            next_contact = False

        self.graph.nodes[id]['latest_contact'] = next_contact
        # this stores a pointer to the latest contact process of this id OR FALSE IF NONE IS SCHEDULED
        # it can be used to interrupt said process should the patient recover in the meantime

    def recover(self, time, id):
        # cancel related contact event

        try:
            if self.graph.nodes[id]['latest_contact']:
                self.cancel_event(id, CONTACT, all=False)
        except KeyError:
            pass

        self.count += inf2rec

        self.update_state(id, REC_STATE)  # individuum is saved as recovered
        self.colormap[id] = 'grey'
        # print(str(id)+' has recovered.')
        # print('Contact process stopped due to recovery.')

    def quarantine(self, time, id):

        connections = list(((id, friend) for friend in self.graph.neighbors(id)))
        for id, friend in connections:
            self.graph.edges[id, friend]['blocked'] = True

        # need to remember the old state
        self.graph.nodes[id]['shadowed_state'] = self.graph.nodes[id]['state']

        # in my simple model it would be possible for someone to be already recovered when the quarantine event happens
        # in this case, the color won't change to blue (because no contact event will ever happen anyways)
        if self.graph.nodes[id]['state'] == REC_STATE:
            pass
        else:
            self.update_state(id, NO_TRANS_STATE)  # update state to transmission disabled
            self.colormap[id] = 'blue'

        heapq.heappush(self.event_list, (time + quarantine_time, END_OF_QUARANTINE, id))

    def end_of_quarantine(self, time, id):

        # if quarantine is over and no state change has happened, it simply gets old one
        if self.graph.nodes[id]['state'] == NO_TRANS_STATE:
            self.update_state(id, self.graph.nodes[id]['shadowed_state'])

        connections = list(((id, friend) for friend in self.graph.neighbors(id)))
        for id, friend in connections:
            if self.graph.nodes[friend]['state'] != NO_TRANS_STATE:
                # if self.colormap[friend] != 'blue':
                # this should keep connections blocked if the other side is in quarantine
                self.graph.edges[id, friend]['blocked'] = False

            #  One would think the last if clause is not necessary...
            #  But otherwise it leaves a weird possibility: if person a and b both are quarantined,
            #  the first one (say a) going out of quarantine would also re-enable the connection between
            #  the two, even if b is still quarantined.

    def tracing(self, time, id):
        contacts = self.graph.nodes[id]['contacts']
        for contact in contacts:
            t_t_random = np.random.exponential(scale=t_t, size=1)[0]
            heapq.heappush(self.event_list, (time + t_t_random, QUARANTINE, contact))
        contacts.clear()

    # simulation

    def sim(self, seed, mode=None):
        # call first infection event

        start = time.time()
        np.random.seed(seed)
        print('Simulation started.')

        event = (0, INFECTION, 0)  # ind. #0 is infected at t = 0
        heapq.heappush(self.event_list, event)

        counter = 0

        while self.event_list:

            event = heapq.heappop(self.event_list)

            current_t = event[0]

            if current_t > self.max_t:
                break

            # if it exceeds the current sampling point, the current counts are saved before doing the event (hold)
            if current_t >= counter * resolution:
                assert (self.count >= 0).all() and self.count.sum() == self.n, \
                    'Something went wrong, impossible states detected.'

                self.counts[:, counter] = self.count
                self.net_states.append((0, self.colormap.copy()))
                counter += 1

            self.do_event(event, mode)

        for i in np.arange(start=counter, stop=self.counts.shape[1], dtype=int):
            self.counts[:, i] = self.counts[:, i - 1]  # otherwise it is all 0 at some point

        end = time.time()
        print('Simulation complete. Simulation time : {}s.'.format(end - start))

        return self.counts

    def do_event(self, event, mode):
        # events are saved as tuples (time, type, id)
        time = event[0]
        type = event[1]
        id = event[2]
        # events:
        # 0:infection
        # 1:infectious
        # 2:contact
        # 3:recovery
        # 4:QUARANTINE
        # 5:END_OF_QUARANTINE
        # 6:TRACING

        if type == 0:
            self.infection(time, id)
        elif type == 1:
            self.infectious(time, id, mode)
        elif type == 2:
            self.contact(time, id)
        elif type == 3:
            self.recover(time, id)
        elif type == QUARANTINE:
            self.quarantine(time, id)
        elif type == END_OF_QUARANTINE:
            self.end_of_quarantine(time, id)
        elif type == TRACING:
            self.tracing(time, id)
        else:
            raise Exception('This event type has not been implemented')

    def cancel_event(self, id, event_id, all=False):
        # the "all" parameter is here because for now I assume that all infection events must be canceled once
        #  the infection is completed.
        #  (so for an infected individual no duplicate infection events occur)
        copy = self.event_list.copy()

        fitting_events = []
        for i, event in enumerate(copy):
            if event[0] == event_id and event[2] == id:
                fitting_events.append((event[0], i))
                # with time and index i have all information needed to cancel
                # NEXT scheduled event with this id and type

        if all:  # want to delete all events of that type for that individual
            indices = [i for bin, i in fitting_events]  # i want these gone
            # https://stackoverflow.com/a/32744088 for using numpy to delete certain entries:

            # now i want to delete the entries that need to be canceled from the list:
            indices.reverse()
            # traverse backwards because deleting the i-th entry would change the following indices
            # NOTE: originally, they are ascending because of enumerate

            for i in indices:
                idx = indices[-i - 1]
                heap_delete(self.event_list, idx)
                #  this might actually be slower than just using del here and heapify in the end
                #  Or would it be both O(n)?

            # this fct is O(n) and by using siftdown and siftup each time i delete an entry i could make it faster, O(logn)
            # however, I have to traverse the whole list anyways at the start so it will always be O(n)...
            return

        else:  # # want to delete just next event of that type for that individual
            # TODO this might not be efficient
            cancel_prioritized = sorted(fitting_events, key=lambda x: x[0])  # sort for time
            try:
                i = cancel_prioritized[0][1]  # gets index of original heap
                heap_delete(self.event_list, i)
            except IndexError:  # no scheduled event that fits
                pass

    def monte_carlo(self, n, mode=None):
        # net is input
        # run sim n times, saving the output in list
        results: List[np.ndarray] = []
        for i in range(n):
            redo = not bool((i + 1) % redo_net)  # redo_net is in globals.py, every i iterations net is changed as well
            self.reset(hard=redo)
            if redo:
                print(self.clustering())
            results.append(self.sim(seed=i, mode=mode).copy())

        # compute mean
        mean = np.zeros(results[0].shape)
        for counts in results:
            mean += counts
        mean /= len(results)

        variance = np.zeros(results[0].shape)
        for counts in results:
            variance += np.square(counts-mean)
        variance /= len(results)

        sd = np.sqrt(variance)

        return mean, sd

    def reset(self, hard=False):
        # see note in __init__. Short: reset to original state (deepcopy), OR redo whole network

        if hard:
            self.__init__(self.n, self.p, self.p_i, self.max_t, self.last_seed + 1,
                          clustering_target=self.clustering_target)
            # this overwrites the network with a new one of different seed (as opposed to just jumping to save point)
        else:
            for key in self.init_state.keys():
                if key != 'init_state':
                    try:
                        self.__dict__[key] = self.init_state[key].copy()
                    except AttributeError:
                        self.__dict__[key] = self.init_state[key]

    # visuals

    def plot_timeseries(self, counts=None, sd=None, save=None, discrete_plots=False, existing_ax=None):
        print('Plotting time series...')

        if not existing_ax:
            # in case of an existing ax, I cannot call these functions or they create a new ax
            plt.clf()
            plt.legend(['susceptible', 'exposed', 'infected', 'recovered'])


        # TODO the discrete view looks bad- haven't found a nice way to
        #  visualize the 0th order spline /hold
        # from https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html

        ts = np.arange(start=0, stop=self.max_t, step=resolution)
        x = ts

        # by default, i use the class's last simulation results.
        # but for monte carlo i want to be able to plot something manually as well
        if isinstance(counts, np.ndarray):
            y = counts.T
        else:
            y = self.counts.T  # in case counts is not given, take the ones saved from last simulation

        ax:matplotlib.axes.Axes = existing_ax or plt.gca()
        ax.set_prop_cycle(color=['green', 'yellow', 'red', 'grey'])  # same as net colormap

        # ax.set_aspect((ax.get_ylim()[1]-ax.get_ylim()[0])*16/((ax.get_xlim()[1]-ax.get_xlim()[0])*9))

        if discrete_plots:
            f = interp1d(x, y, kind='zero', axis=0)
            x_new = np.linspace(0, self.max_t - resolution, num=10001, endpoint=False)

            ax.plot(x_new, f(x_new))

            assert not isinstance(sd, np.ndarray), \
                "discrete view does not make much sense for standard deviation, " \
                "not implemented"

        else:
            ax.plot(x, y)

            if isinstance(sd, np.ndarray):
                ax.plot(x,y+sd.T, '--', x,y-sd.T, '--')


        if save:
            assert not ax, 'save = True is not supported with an input ax. ' \
                           'The latter is only for plotting something as part of a bigger function'
            plt.savefig(save)
        elif ax:
            return ax
        else:
            plt.show()

    def draw(self):
        pos = self.pos
        # i deliberately leave the seed fixed, maybe I want same positions for networks of equal size
        nx.draw(self.graph, node_color=self.colormap, pos=pos)

        plt.show()

    def animate_last_sim(self, dest=None):
        print("Generating animation...")
        start = time.time()

        assert self.net_states, "You need to run the simulation first!"
        matplotlib.interactive(False)
        fig = plt.figure()
        pos = self.pos

        nodes = nx.draw_networkx_nodes(self.graph, pos, node_color=self.net_states[0][1], node_size=3)
        edges = nx.draw_networkx_edges(self.graph, pos, width=0.1)

        # function that draws a single frame from a saved state
        def animate(idx):
            nodes.set_color(self.net_states[idx][1])
            # edges = nx.draw_networkx_edges(self.graph, pos, width=0.1)

            return nodes,

        anim = animation.FuncAnimation(fig, animate, frames=len(self.net_states), interval=1000, blit=False)

        # save to specified dir or just in working dir
        if not dest:
            anim.save('last_vid.mp4')
        else:
            anim.save(dest)
        plt.close(fig)

        end = time.time()
        print('Saved animation. Time elapsed: {}s.'.format(end - start))

    # convenience:

    def update_state(self, id, state):
        self.graph.nodes[id]['state'] = state

    # misc

    def clustering(self):
        return nx.average_clustering(self.graph)

    def alter_clustering_coeff(self, target, epsilon):
        # to make less homogenous networks, this function redistributes edges until sufficiently close to goal

        current_coeff = nx.average_clustering(self.graph)

        budget = 10000 * self.n


        check_skipping = self.n/5
        # This should depend on n since for smaller networks each swapped edge is weighted heavier
        counter = 0
        stage = 0 # try several different check skipping values, maybe convergence is too fast/too slow

        # the epsilon tolerance is relative to p, the normal clustering coeff in a random network
        while abs(current_coeff - target) > epsilon*self.p and counter < budget:
            a, b = np.random.randint(0, high=self.n, size=2, dtype=int)

            neighbors_a = list(self.graph.neighbors(a))
            neighbors_b = list(self.graph.neighbors(b))

            if target > current_coeff:
                # currently coeff is too low

                if len(neighbors_a) > len(neighbors_b):
                    # a gets edge from b to increase coeff
                    c = np.random.choice(neighbors_b)
                    if (not c in neighbors_a) and len(neighbors_b) != 1:
                        # only move an edge when no edge between new partners exist AND at least 1 edge would be left

                        self.graph.remove_edge(b, c)
                        self.graph.add_edge(a, c)
                        # current_coeff = nx.average_clustering(self.graph)
                        if counter % check_skipping == 0:
                            # print(current_coeff)
                            # print(len(self.graph.edges))
                            current_coeff = nx.average_clustering(self.graph)  # heuristic, do it in batches
                else:
                    # b gets edge from a
                    c = np.random.choice(neighbors_a)
                    if (not c in neighbors_b) and len(neighbors_a) != 1:
                        # only move an edge when no edge between new partners exist AND at least 1 edge would be left
                        self.graph.remove_edge(a, c)
                        self.graph.add_edge(b, c)
                        # current_coeff = nx.average_clustering(self.graph)
                        if counter % 100 == 0:  # 100 is just an idea
                            current_coeff = nx.average_clustering(self.graph)  # heuristic, do it in batches

            else:
                # coeff is too high
                if len(neighbors_b) > len(neighbors_a):  # This is the only different line, everything is flipped
                    # a gets edge from b to increase coeff
                    c = np.random.choice(neighbors_b)
                    if (not c in neighbors_a) and len(neighbors_b) != 1:
                        # only move an edge when no edge between new partners exist AND at least 1 edge would be left
                        self.graph.remove_edge(b, c)
                        self.graph.add_edge(a, c)
                        # current_coeff = nx.average_clustering(self.graph)
                        if counter % 100 == 0:  # 100 is just an idea
                            current_coeff = nx.average_clustering(self.graph)  # heuristic, do it in batches
                else:
                    # b gets edge from a
                    c = np.random.choice(neighbors_a)
                    if (not c in neighbors_b) and len(neighbors_a) != 1:
                        # only move an edge when no edge between new partners exist AND at least 1 edge would be left
                        self.graph.remove_edge(a, c)
                        self.graph.add_edge(b, c)
                        # current_coeff = nx.average_clustering(self.graph)
                        if counter % 100 == 0:  # 100 is just an idea
                            current_coeff = nx.average_clustering(self.graph)  # heuristic, do it in batches

            counter += 1
            if counter == budget:
                if stage == 0:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping /= 4
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_coeff))
                    continue
                elif stage == 1:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping *= 16
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_coeff))
                    continue



        assert (counter != budget), "no success in changing clustering coefficient accordingly"

        return current_coeff


if __name__ == '__main__':
    p_i = 0.9
    net = Net(n=100, p_i=p_i, p=0.3, seed=123, max_t=100)
    # net.draw()

    test1 = net.sim(seed=123, mode='quarantine')
    # test2 = net.sim(seed=123, mode='tracing')

    net.plot_timeseries()

    # print(np.all(test1 == test2))

    # print(net.alter_clustering_coeff(0.09, 0.001))

    # net.animate_last_sim()
