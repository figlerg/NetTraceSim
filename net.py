# network is a class that implements the full simulation environment
import copy

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import heapq
import matplotlib.animation as animation
import time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
from typing import List


from globals import *  # loading some variables and constants
from helpers import heap_delete, disp


class Net(object):

    def __init__(self, n, p, p_i, max_t, seed, clustering_target=None,
                 dispersion_target = None):



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
        self.dispersion_target = dispersion_target # dispersion

        self.event_list = []  # create event list as list and heapify for priority queue
        heapq.heapify(self.event_list)

        #for dumping the event history (verification)
        self.event_history = []


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

        # for comparison, even new network structures use same layout (this only writes self.pos once, at start)
        if not hasattr(self, 'pos'):
            self.pos = nx.spring_layout(self.graph, seed=seed)



        assert not (clustering_target and dispersion_target), "Cannot set a dispersion target and " \
                                                            "a clustering target at the same time"

        if self.clustering_target:
            self.final_cluster_coeff = self.alter_clustering_coeff(clustering_target, epsilon_clustering)
        else:
            self.final_cluster_coeff = self.clustering()
        if self.dispersion_target:
            self.final_dispersion = self.alter_disp(dispersion_target, epsilon_disp)
        else:
            self.final_dispersion = disp(self.graph)

        # I don't want to deal with a whole mutable state list, so I only save the current count at regular intervals:
        self.count = np.zeros([4, 1], dtype=np.int32).flatten()  # current state
        # susceptible, exposed, infectious, recovered are the 4 rows

        # history, gets written in sim()
        self.count[0] = n
        self.counts = np.zeros([4, math.floor(max_t / resolution)], dtype=np.int32)

        self.net_states = []  # this is a list of nets at equidistant time steps
        # i use this for the animation frames



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

            # in some cases the infection isn't noticed
            u = np.random.uniform()
            if u < p_q:
                heapq.heappush(self.event_list, (time + t_q_random, QUARANTINE, id))

            if mode == 'tracing':

                # if infection isn't noticed, no tracing is issued (same u of course)
                if u < p_q:
                    heapq.heappush(self.event_list, (time + t_q_random, TRACING, id))
                    # I will simply do these two at the same time (when the infection is detected)
                    # the tracing event adds a little bit of time for the process of finding and alerting contacts

    def contact(self, time, id):

        friends = list((friend for friend in self.graph.neighbors(id)
                        if self.graph.edges[id, friend]['blocked'] == False))
        # can only use edges that aren't blocked due to quarantine

        if not friends:
            # just in case this node is isolated right now, it should still try contacting people later until it recovered
            t_c_random = np.random.exponential(scale=t_c, size=1)[0]
            next_contact = (time + t_c_random, CONTACT, id)
            heapq.heappush(self.event_list, next_contact)
            return

        for contacted_friend in friends:
            self.graph.nodes[id]['contacts'].append(contacted_friend)

            if self.graph.nodes[contacted_friend]['state'] == SUSC_STATE:

                # print('#' + str(id) + ' has had contact with #{}.'.format(contacted_friend))
                u = np.random.uniform()

                if u < self.p_i:
                    heapq.heappush(self.event_list, (time, INFECTION, contacted_friend))
            else:
                pass  # if in any other state than susceptible, this contact does not matter

        # if friends:
        #     contacted_friend_idx = np.random.choice(len(friends), 1)[0]
        #     contacted_friend = friends[contacted_friend_idx]
        #     self.graph.nodes[id]['contacts'].append(contacted_friend)
        # else:
        #     t_c_random = np.random.exponential(scale=t_c, size=1)[0]
        #     next_contact = (time + t_c_random, CONTACT, id)
        #     heapq.heappush(self.event_list, next_contact)
        #     return
        #
        # if self.graph.nodes[contacted_friend]['state'] == SUSC_STATE:
        #
        #     # print('#' + str(id) + ' has had contact with #{}.'.format(contacted_friend))
        #     u = np.random.uniform()
        #
        #     if u < self.p_i:
        #         heapq.heappush(self.event_list, (time, INFECTION, contacted_friend))
        # else:
        #     pass  # if in any other state than susceptible, this contact does not matter

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

            if np.random.uniform() < p_t:
                heapq.heappush(self.event_list, (time + t_t_random, QUARANTINE, contact))
        contacts.clear()

    # simulation

    def sim(self, seed, mode=None):
        # call first infection event

        start = time.time()
        np.random.seed(seed)
        print('Simulation started.')

        event = (0, INFECTION, 0)  # ind. #0 is infected at t = 0
        event2 = (0, INFECTION, 1) # several patients to make the start more stable
        event3 = (0, INFECTION, 2)
        heapq.heappush(self.event_list, event)
        heapq.heappush(self.event_list, event2)
        heapq.heappush(self.event_list, event3)


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

        self.event_history.append(event)

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
            if event[1] == event_id and event[2] == id:
                fitting_events.append((event[0], i))
                # with time and index i have all information needed to cancel
                # NEXT scheduled event with this id and type

        if all:  # want to delete all events of that type for that individual
            indices = [i for bin, i in fitting_events]  # i want these gone
            # https://stackoverflow.com/a/32744088 for using numpy to delete certain entries:

            # now i want to delete the entries that need to be canceled from the list:
            # traverse backwards because deleting the i-th entry would change the following indices
            # TODO this still deletes the wrong events
            for i in reversed(indices):
                heap_delete(self.event_list, i)
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
        results_peaks: List[float] = []
        results_prevalence: List[float] = []
        net_cluster_coeffs: List[float] = []
        net_disps : List[float] = []
        for i in range(n):
            redo = not bool((i + 1) % redo_net)  # redo_net is in globals.py, every i iterations net is changed as well
            self.reset(hard=redo)
            # if redo:
            #     print(self.clustering())
            counts = self.sim(seed=i, mode=mode).copy()
            results.append(counts)
            net_cluster_coeffs.append(self.final_cluster_coeff)
            net_disps.append(self.final_dispersion)

            # compute when the peak happens and what the ratio of infected is then
            exposed = counts[EXP_STATE, :]
            infected = counts[INF_STATE, :]
            ep_curve = exposed + infected
            t_peak = np.argmax(ep_curve, axis=0)
            peak_height = ep_curve[t_peak] /self.n
            results_peaks.append(peak_height)

            # period prevalence
            recovered = counts[REC_STATE, :]
            virus_contacts = ep_curve + recovered
            period_prevalence = virus_contacts[-1] / n
            results_prevalence.append(period_prevalence)


        # compute mean
        mean_counts = np.zeros(results[0].shape)
        meansq_counts = np.zeros(results[0].shape)
        for counts in results:
            mean_counts += counts
            meansq_counts += np.square(counts)
        mean_counts /= len(results)
        meansq_counts /= len(results)

        mean_peaks = np.mean(results_peaks)
        meansq_peaks = np.mean(np.square(results_peaks))

        mean_prevalence = np.mean(results_prevalence)
        meansq_prevalence = np.mean(np.square(results_prevalence))

        mean_clustering = np.mean(net_cluster_coeffs)
        mean_disp = np.mean(net_disps)

        return mean_counts, meansq_counts, mean_peaks, meansq_peaks, mean_prevalence, meansq_prevalence , mean_clustering, mean_disp

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

    def draw(self, show=True):
        pos = self.pos
        # i deliberately leave the seed fixed, maybe I want same positions for networks of equal size
        # nx.draw(self.graph, node_color=self.colormap, pos=pos)

        nodes = nx.draw_networkx_nodes(self.graph, pos, node_size=1)
        edges = nx.draw_networkx_edges(self.graph, pos, width=0.05)

        if show:
                plt.show()
        else:
            return

    def animate_last_sim(self, dest=None):
        print("Generating animation...")
        start = time.time()

        assert self.net_states, "You need to run the simulation first!"
        matplotlib.interactive(False)
        fig = plt.figure()
        pos = self.pos

        nodes = nx.draw_networkx_nodes(self.graph, pos, node_color=self.net_states[0][1], node_size=40)
        edges = nx.draw_networkx_edges(self.graph, pos, width=0.4)

        # function that draws a single frame from a saved state
        def animate(idx):
            nodes.set_color(self.net_states[idx][1])
            # edges = nx.draw_networkx_edges(self.graph, pos, width=0.1)

            if idx == 50:
                nx.draw(self.graph)

            return nodes,



        anim = animation.FuncAnimation(fig, animate, frames=len(self.net_states), interval=1000, blit=False)

        # save to specified dir or just in working dir
        if not dest:
            anim.save('last_vid.gif')
        else:
            anim.save(dest, dpi=1000)
        plt.close(fig)

        end = time.time()
        print('Saved animation. Time elapsed: {}s.'.format(end - start))

    def parse_event_history(self):
        out = []
        for event in self.event_history:
            out.append(list(event))

        # rename the type ints for better reading
        for event in out:
            type = event[1]
            str_map = {0:'Infection', 1:'Infectious', 2:'Contact', 3:'Recover',4:'Quarantine',
                       5:'End of Tracing', 6:'Tracing'}

            event[1] = str_map[type]

        out = pd.DataFrame(out, columns=['t','Type','ID'])


        return out

    # convenience:

    def update_state(self, id, state):
        self.graph.nodes[id]['state'] = state

    # misc

    def clustering(self):
        return nx.average_clustering(self.graph)

    def dispersion(self):
        return disp(self.graph)

    def alter_clustering_coeff(self, target, epsilon):
        # to make less homogenous networks, this function redistributes edges until sufficiently close to goal

        current_coeff = nx.average_clustering(self.graph)

        budget = 10000 * self.n


        check_skipping = self.n/2
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

            # trying some bigger and smaller batch sizes
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
                elif stage == 2:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping = check_skipping/4 *10
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_coeff))
                    continue
                elif stage == 3:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping = check_skipping/10/10
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_coeff))
                    continue



        assert (counter != budget), "no success in changing clustering coefficient accordingly"

        return current_coeff

    def alter_disp(self, target, epsilon):
        # to make less homogenous networks, this function redistributes edges until sufficiently close to goal

        current_disp = disp(self.graph)

        budget = 10000 * self.n
        check_skipping = self.n/10
        # This should depend on n since for smaller networks each swapped edge is weighted heavier

        counter = 0
        stage = 0 # try several different check skipping values, maybe convergence is too fast/too slow

        # the epsilon tolerance is relative to p, the normal clustering coeff in a random network
        while abs(current_disp - target) > epsilon and counter < budget:
            a, b = np.random.randint(0, high=self.n, size=2, dtype=int)

            neighbors_a = list(self.graph.neighbors(a))
            neighbors_b = list(self.graph.neighbors(b))

            if target > current_disp:
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
                            # print(current_disp)
                            # print(len(self.graph.edges))
                            current_disp = disp(self.graph)
                            # current_coeff = nx.average_clustering(self.graph)  # heuristic, do it in batches
                else:
                    # b gets edge from a
                    c = np.random.choice(neighbors_a)
                    if (not c in neighbors_b) and len(neighbors_a) != 1:
                        # only move an edge when no edge between new partners exist AND at least 1 edge would be left
                        self.graph.remove_edge(a, c)
                        self.graph.add_edge(b, c)
                        # current_coeff = nx.average_clustering(self.graph)
                        if counter % 100 == 0:  # 100 is just an idea
                            # current_coeff = nx.average_clustering(self.graph)  # heuristic, do it in batches
                            current_disp = disp(self.graph)


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
                            current_disp = disp(self.graph)

                else:
                    # b gets edge from a
                    c = np.random.choice(neighbors_a)
                    if (not c in neighbors_b) and len(neighbors_a) != 1:
                        # only move an edge when no edge between new partners exist AND at least 1 edge would be left
                        self.graph.remove_edge(a, c)
                        self.graph.add_edge(b, c)
                        # current_coeff = nx.average_clustering(self.graph)
                        if counter % 100 == 0:  # 100 is just an idea
                            current_disp = disp(self.graph)

            counter += 1

            # trying some bigger and smaller batch sizes
            if counter == budget:
                if stage == 0:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping /= 4
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_disp))
                    continue
                elif stage == 1:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping *= 16
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_disp))
                    continue
                elif stage == 2:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping = check_skipping/4 *10
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_disp))
                    continue
                elif stage == 3:
                    print('Having difficulties reaching clustering target- changing skipping constant')
                    check_skipping = check_skipping/10/10
                    counter = 0
                    stage += 1
                    print('target:{}, val:{}'.format(target,current_disp))
                    continue


        # print('failed:{}'.format(current_disp-target))
        assert (counter != budget), "no success in changing clustering coefficient accordingly"

        return current_disp


if __name__ == '__main__':
    p_i = 0.9
    net = Net(n=100, p=0.3, p_i=p_i, max_t=100, seed=123)
    # net.draw()

    test1 = net.sim(seed=123, mode='quarantine')
    # test2 = net.sim(seed=123, mode='tracing')

    net.plot_timeseries()

    # print(np.all(test1 == test2))

    # print(net.alter_clustering_coeff(0.09, 0.001))

    # net.animate_last_sim()
