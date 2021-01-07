# try to use whole network as class that runs in environment

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
import time
from scipy.interpolate import interp1d

from globals import *  # loading some variables and constants


class Net(object):

    def __init__(self, n, p, max_t, seed):

        # TODO try and decrease complexity, this seems convoluted

        print("Initializing network...")

        start = time.time()

        np.random.seed(seed)
        # random.seed(seed)

        self.n = n

        # self.graph = nx.gnp_random_graph(n, p, seed = seed)
        self.graph = nx.fast_gnp_random_graph(n, p, seed = seed)

        self.colormap = ['green' for i in range(n)]

        self.event_list = []
        heapq.heapify(self.event_list)

        self.max_t = max_t

        # I dont want to deal with a whole mutable state list, so I only save the current count at regular intervals:
        self.count = np.zeros([5,1], dtype=np.int32).flatten() # current state
        # susceptible, exposed, infectious, recovered, transmission_disabled are the 5 rows

        self.count[0] = n
        self.counts = np.zeros([5, max_t//resolution], dtype=np.int32) # history, gets written in sim()



        self.net_states = [] # this is a list of nets at equidistant time steps
        # i use this for the animation
        self.pos = nx.spring_layout(self.graph, seed=seed)

        for id in range(n):
            # at first all are susceptible
            # print(net.nodes)
            # print(net.edges)
            self.graph.nodes[id]['state'] = 0

        # TODO this is a little rough...
        #  essentially I want to set a reset point because some of the values are changed in place and I need a fresh
        #  start for each monte carlo. Resetting is done via the self.reset() function
        self.init_state = {}
        for key in self.__dict__.keys():
            try:
                self.init_state[key] = self.__dict__[key].copy()
            except AttributeError:
                # print(key)
                self.init_state[key] = self.__dict__[key]

        end = time.time()

        print("Network initialized. Time elapsed: {}s.".format(end - start))

        # self.draw()
    # events:

    def infection(self, time, id):

        self.update_state(id, 1)  # exposed now
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
        self.update_state(id,2)
        self.count += exp2inf
        self.colormap[id] = 'red'

        t_c_random = np.random.exponential(scale=t_c, size=1)[0]
        t_r_random = np.random.exponential(scale=t_r, size=1)[0]



        heapq.heappush(self.event_list, (time + t_c_random,CONTACT, id))
        heapq.heappush(self.event_list, (time + t_r_random ,RECOVER, id))

        if mode == 'quarantine':
            t_q_random = np.random.exponential(scale=t_q, size=1)[0]
            heapq.heappush(self.event_list, (time + t_q_random ,QUARANTINE, id))


    def contact(self, time, id):

        friends = list(self.graph.neighbors(id))
        if friends:
            # contacted_friend = random.choice(friends)
            contacted_friend_idx = np.random.choice(len(friends),1)[0]
            contacted_friend = friends[contacted_friend_idx]
        else:
            return

        # if self.graph.nodes[id]['state'] == 3:
        #     yield

        if self.graph.nodes[contacted_friend]['state'] == 0:

            # print('#' + str(id) + ' has had contact with #{}.'.format(contacted_friend))
            # u = random.uniform(0,1)
            u = np.random.uniform()

            if u < p_i:
                heapq.heappush(self.event_list, (time, INFECTION, contacted_friend))
        else:
            pass  # if in any other state than susceptible, this contact does not matter




        if self.graph.nodes[id]['state'] == 2:

            t_c_random = np.random.exponential(scale=t_c, size=1)[0]

            next_contact = (time+t_c_random, CONTACT, id)
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
        # try:
        #     if self.graph.nodes[id]['latest_contact']:
        #         copy = self.event_list.copy()
        #
        #         # TODO this could be faster, i can use heap structure to stop earlier right?
        #         fitting_events = []
        #         for i, event in enumerate(copy):
        #             if event[0] == 2 and event[2] == id:
        #                 fitting_events.append((event[0], i))
        #                 # with time and index i have all information needed to cancel
        #                 # NEXT scheduled event with this id and type
        #         cancel_prioritized = sorted(fitting_events, key= lambda x: x[0]) # sort for time
        #         try:
        #             i = cancel_prioritized[0][1] # gets index of original heap
        #             heap_delete(self.event_list, i)
        #         except IndexError: # no scheduled event that fits
        #             pass
        # except:
        #     pass

        if self.graph.nodes[id]['state'] == NO_TRANS_STATE:
            self.count += no_trans2rec
        else:
            self.count += inf2rec

        self.update_state(id, 3)  # individuum is saved as recovered
        self.colormap[id] = 'grey'
        # print(str(id)+' has recovered.')

        # print('Contact process stopped due to recovery.')


    def quarantine(self, time, id):

        # in my simple model it would be possible for someone to be already recovered when the quarantine event happens
        # in this case, no quarantine is necessary (and it would not change anything anymore)
        # TODO think about this logic
        if self.graph.nodes[id]['state'] == REC_STATE:
            return

        self.update_state(id, NO_TRANS_STATE)  # update state to transmission disabled
        self.count += inf2no_trans
        self.colormap[id] = 'blue'

        heapq.heappush(self.event_list, (time + quarantine_time,CONTACT, id))



    def end_of_quarantine(self, time, id):
        # TODO check whether this makes sense. Could also simply put them into recovered in the end.
        #  also, probably one would do a test at end of quarantine?
        if self.graph.nodes[id]['state'] == NO_TRANS_STATE:
            # no recovery happened, so back to infectious.
            self.update_state(id, INF_STATE)
            t_c_random = np.random.exponential(scale=t_c, size=1)[0]
            heapq.heappush(self.event_list, (time + t_c_random,CONTACT, id))
            self.count += no_trans2rec
        elif self.graph.nodes[id]['state'] == REC_STATE:
            # already recovered
            return
        else:
            raise Exception("Something went wrong, an end of quarantine was scheduled for id with invalid state")


    # simulation

    def sim(self, seed, animation=False, mode = None):
        # call first infection event

        np.random.seed(seed)

        start = time.time()

        print('Simulation started.')

        event = (0, INFECTION, 0)  # ind. #0 is infected at t = 0
        heapq.heappush(self.event_list, event)

        # intervals = 1 # days for each animation frame
        counter = 0

        # end_of_sim = -1

        while self.event_list:

            event = heapq.heappop(self.event_list)

            current_t = event[0]

            if current_t > self.max_t:
                break

            # if it exceeds the current sampling point, the current counts are saved before doing the event (hold)
            if current_t >= counter * resolution:
                assert (
                                   self.count >= 0).all() and self.count.sum() == self.n, 'Something went wrong, impossible states detected.'

                self.counts[:, counter] = self.count
                self.net_states.append((0, self.colormap.copy()))
                counter += 1

            self.do_event(event, mode)

        end_of_sim = current_t  # this is where the simulation stopped. After that, states remain constant
        for i in np.arange(start=counter, stop=self.counts.shape[1], dtype=int):
            self.counts[:, i] = self.counts[:, i - 1]  # otherwise it is all 0 at some point

        end = time.time()

        print('Simulation complete. Simulation time : {}s.'.format(end - start))

        # self.plot_timeseries()
        return self.counts

    def do_event(self, event, mode):
        time = event[0]
        type = event[1]  # REARRANGED as (time, type, id) because heapq sorts only for first...
        # TODO check whether i changed it everywhere...
        id = event[2]
        # events:
        # 0:infection
        # 1:infectious
        # 2:contact
        # 3:recovery

        if type == 0:
            self.infection(time, id)
        elif type == 1:
            self.infectious(time, id, mode=mode)
        elif type == 2:
            self.contact(time, id)
        elif type == 3:
            self.recover(time, id)
        elif type == QUARANTINE:
            self.quarantine(time, id)
        elif type == END_OF_QUARANTINE:
            self.end_of_quarantine(time,id)
        else:
            raise Exception('This event type has not been implemented')

    def cancel_event(self, id, event_id, all=False):
        # the "all" parameter is here because for now I assume that all infection events must be canceled once
        #  the infection has commenced TODO review
        #  (so for an infected individual no other infection events shall occur
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
            # copy = np.delete(copy, indices).tolist()

            # now i want to delete the entries that need to be canceled from the list:
            indices.reverse()
            # traverse backwards because deleting the i-th entry would change the following indices
            # NOTE: originally, they are ascending because of enumerate

            for i in indices:
                idx = indices(-i - 1)
                heap_delete(self.event_list, idx)
                # TODO this might actually be worse than just using del here and heapify in the end
                #  I think this would be both O(n)?

            # this is O(n) and by using siftdown and siftup each time i delete an entry i could make it faster, O(logn)
            # however, I have to traverse the whole list anyways at the start so it will always be O(n)...
            # heapq.heapify(copy)
            # self.event_list = copy
            return

        else:  # # want to delete just next event of that type for that individual
            # TODO this is not efficient
            cancel_prioritized = sorted(fitting_events, key=lambda x: x[0])  # sort for time
            try:
                i = cancel_prioritized[0][1]  # gets index of original heap
                heap_delete(self.event_list, i)
            except IndexError:  # no scheduled event that fits
                pass

    def reset(self):
        # see note in __init__. Short: reset to original state (deepcopy)
        # TODO unsafe?
        for key in self.init_state.keys():
            if key != 'init_state':
                try:
                    self.__dict__[key] = self.init_state[key].copy()
                except AttributeError:
                    # print(key)
                    self.__dict__[key] = self.init_state[key]

    # visuals

    def plot_timeseries(self, counts=None):
        print('Plotting time series...')

        n_counts = self.counts.shape[1]
        ts = np.arange(start=0, stop=self.max_t, step=resolution)

        # TODO this is not optimal... I would like the vertical lines to disappear
        # from https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
        # x = np.linspace(0, 10, num=11, endpoint=True)
        x = ts

        # by default, i use the classes last simulation results.
        # but for monte carlo i want to be able to plot something manually as well
        if isinstance(counts, np.ndarray):
            y = counts.T
        else:
            y = self.counts.T  # in case counts is not given, take the ones saved from last simulation

        # f1 = interp1d(x, y, kind='nearest')
        f2 = interp1d(x, y, kind='previous', axis=0)
        # f3 = interp1d(x, y, kind='next')
        xnew = np.linspace(0, self.max_t - resolution, num=10001, endpoint=False)
        # plt.plot(x, y, 'o')
        plt.plot(xnew, f2(xnew))
        # plt.legend(['data', 'nearest', 'previous', 'next'], loc='best')
        plt.show()

        # plt.plot(ts, self.counts.T)
        # plt.show()

    def draw(self):
        pos = self.pos
        # i deliberately leave the seed fixed, maybe I want same positions for networks of equal size
        nx.draw(self.graph, node_color = self.colormap, pos = pos)

        plt.show()

    def animate_last_sim(self):
        print("Generating animation...")
        start = time.time()

        assert self.net_states, "You need to run the simulation first!"

        fig = plt.figure()
        pos = self.pos

        nodes = nx.draw_networkx_nodes(self.graph, pos, node_color=self.net_states[0][1], node_size=3)
        edges = nx.draw_networkx_edges(self.graph, pos, width=0.1)

        # function that draws a single frame from a saved state
        def animate(idx):
            nodes.set_color(self.net_states[idx][1])
            edges = nx.draw_networkx_edges(self.graph, pos, width=0.1)

            return nodes,
            # net_state = self.net_states[idx]

            # graph = net_state[0]
            # colormap = net_state[1]
            # fig.clf()

            # nx.draw(graph, node_color = colormap, pos = pos)

        anim = animation.FuncAnimation(fig, animate, frames=len(self.net_states), interval=1000, blit=False)

        anim.save('test.mp4')

        end = time.time()
        print('Saved animation. Time elapsed: {}s.'.format(end - start))

    # convenience:

    def update_state(self, id, state):
        self.graph.nodes[id]['state'] = state

def heap_delete(h:list, i):
    # from https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
    # this is O(logn)
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)


if __name__ == '__main__':

    net = Net(n = 100, p = 0.1, seed = 123, max_t=100)
    # net.draw()

    net.sim(seed= 123)




    # net.animate_last_sim()



