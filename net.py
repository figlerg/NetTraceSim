# try to use whole network as class that runs in environment

import networkx as nx
import numpy as np
# from person import Person
import matplotlib.pyplot as plt
import random
import heapq
import matplotlib.animation as animation
import time
from scipy.interpolate import interp1d

p_i = 0.5  # probability of infection at contact
t_i = 2  # incubation time
t_r = 14  # recovery time
t_c = 1  # time between contacts

# NOTE: this is the same as the scale parameter in np.random.exponential! No inverse needed! Checked experimentally...

resolution = 20 # days for each animation frame, abtastrate

INFECTION = 0
INFECTIOUS = 1
CONTACT = 2
RECOVER = 3


# update the counts in the events by adding these to self.count:
susc2exp = np.asarray([-1,1,0,0,0], dtype=np.int32)
exp2inf = np.asarray([0,-1,1,0,0], dtype=np.int32)
inf2rec = np.asarray([0,0,-1,1,0], dtype=np.int32)
inf2no_trans = np.asarray([0,0,-1,0,1], dtype=np.int32)
no_trans2rec = np.asarray([0,0,0,1,-1], dtype=np.int32)



class Net(object):

    def __init__(self, n, p, max_t, seed):

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

        end = time.time()

        print("Network initialized. Time elapsed: {}s.".format(end-start))

        # self.draw()
    # events:

    def infection(self, time, id):

        self.update_state(id,1) # exposed now
        self.count += susc2exp
        self.colormap[id] = 'yellow'
        # print('Person #{} has been exposed at time {}'.format(id, time))


        # schedule infectious event

        t_i_random = np.random.exponential(scale=t_i, size=1)[0]
        heapq.heappush(self.event_list, (time + t_i_random, INFECTIOUS, id))

    def infectious(self, time, id):
        # print('Person #{} started being infectious at time {}'.format(id, time))
        self.update_state(id,2)
        self.count += exp2inf
        self.colormap[id] = 'red'

        t_c_random = np.random.exponential(scale=t_c, size=1)[0]
        t_r_random = np.random.exponential(scale=t_r, size=1)[0]


        heapq.heappush(self.event_list, (time + t_c_random,CONTACT, id))
        heapq.heappush(self.event_list, (time + t_r_random ,RECOVER, id))


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
                copy = self.event_list.copy()

                # TODO this could be faster, i can use heap structure to stop earlier right?
                fitting_events = []
                for i, event in enumerate(copy):
                    if event[0] == 2 and event[2] == id:
                        fitting_events.append((event[0], i))
                        # with time and index i have all information needed to cancel
                        # NEXT scheduled event with this id and type
                cancel_prioritized = sorted(fitting_events, key= lambda x: x[0]) # sort for time
                try:
                    i = cancel_prioritized[0][1] # gets index of original heap
                    heap_delete(self.event_list, i)
                except IndexError: # no scheduled event that fits
                    pass
        except:
            pass


        self.update_state(id,3) # individuum is saved as recovered
        self.count += inf2rec
        self.colormap[id] = 'grey'
        # print(str(id)+' has recovered.')

        # print('Contact process stopped due to recovery.')



# convenience:

    def sim(self, seed, animation = True):
        # call first infection event

        np.random.seed(seed)

        start = time.time()

        print('Simulation started.')


        event = (0, INFECTION, 0) # ind. #0 is infected at t = 0
        heapq.heappush(self.event_list, event)

        # intervals = 1 # days for each animation frame
        counter = 0



        while self.event_list:

            event = heapq.heappop(self.event_list)

            current_t = event[0]

            if current_t > self.max_t:
                break

            # if it exceeds the current sampling point, the current counts are saved before doing the event (hold)
            if current_t >= counter * resolution:
                self.counts[:,counter] = self.count
                self.net_states.append((0, self.colormap.copy()))
                counter += 1

            self.do_event(event)


        end = time.time()


        print('Simulation complete. Simulation time : {}s.'.format(end-start))

        self.plot_timeseries()

    def plot_timeseries(self):
        print('Plotting time series...')
        n_counts = self.counts.shape[1]
        ts = np.arange(start=0, stop=self.max_t, step=resolution)



        # TODO this is not optimal... I would like the vertical lines to disappear
        # from https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
        # x = np.linspace(0, 10, num=11, endpoint=True)
        x = ts
        y = self.counts.T
        # f1 = interp1d(x, y, kind='nearest')
        f2 = interp1d(x, y, kind='previous', axis=0)
        # f3 = interp1d(x, y, kind='next')
        xnew = np.linspace(0, self.max_t-resolution, num=10001, endpoint=False)
        # plt.plot(x, y, 'o')
        plt.plot(xnew, f2(xnew))
        # plt.legend(['data', 'nearest', 'previous', 'next'], loc='best')
        plt.show()

        # plt.plot(ts, self.counts.T)
        # plt.show()



    def do_event(self, event):
        time = event[0]
        type = event[1] # REARRANGED as (time, type, id) because heapq sorts only for first...
        # TODO check whether i changed it everywhere...
        id = event[2]
        # events:
            # 0:infection
            # 1:infectious
            # 2:contact
            # 3:recovery


        if type == 0:
            self.infection(time,id)
        elif type == 1:
            self.infectious(time,id)
        elif type == 2:
            self.contact(time,id)
        elif type == 3:
            self.recover(time, id)
        else:
            raise Exception('This event type has not been implemented')

    def draw(self):
        pos = self.pos
        # i deliberately leave the seed fixed, maybe I want same positions for networks of equal size
        nx.draw(self.graph, node_color = self.colormap, pos = pos)

        plt.show()

    def update_state(self, id, state):
        self.graph.nodes[id]['state'] = state

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
        print('Saved animation. Time elapsed: {}s.'.format(end-start))

    # def animate(self,idx):
    #     pos = nx.spring_layout(self.graph, seed=100)
    #     nx.draw(self.net_states[idx], node_color = self.colormap, pos = pos)

    # REMINDER:

def heap_delete(h:list, i):
    # from https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
    # this is O(logn)
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)


if __name__ == '__main__':

    net = Net(n = 1000, p = 0.1, seed = 123, max_t=100)
    # net.draw()

    net.sim(seed= 123)



    # net.animate_last_sim()



