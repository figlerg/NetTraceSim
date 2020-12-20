# try to use whole network as class that runs in environment

import networkx as nx
import numpy as np
# from person import Person
import matplotlib.pyplot as plt
import random
import heapq
import matplotlib.animation as animation

p_i = 0.5  # probability of infection at contact
t_i = 2  # incubation time
t_r = 14  # recovery time
t_c = 1  # rate of contacts

INFECTION = 0
INFECTIOUS = 1
CONTACT = 2
RECOVER = 3


class Net(object):

    def __init__(self, n, p, seed = 12345):

        self.n = n
        self.graph = nx.gnp_random_graph(n, p, seed = seed)
        self.colormap = ['green' for i in range(n)]
        random.seed(seed)

        self.event_list = []
        heapq.heapify(self.event_list)

        self.net_states = [] # this is a list of nets at equidistant time steps
        # i use this for the animation
        self.pos = nx.spring_layout(self.graph, seed=seed)


        for id in range(n):
            # at first all are susceptible
            # print(net.nodes)
            # print(net.edges)
            self.graph.nodes[id]['state'] = 0

        self.draw()
    # events:

    def infection(self, time, id):

        self.update_state(id,1) # exposed now
        self.colormap[id] = 'yellow'
        print('Person #{} has been exposed at time {}'.format(id, time))


        # schedule infectious event
        heapq.heappush(self.event_list, (time + t_i, INFECTIOUS, id))

    def infectious(self, time, id):
        print('Person #{} started being infectious at time {}'.format(id, time))
        self.update_state(id,2)
        self.colormap[id] = 'red'

        heapq.heappush(self.event_list, (time + t_c,CONTACT, id))
        heapq.heappush(self.event_list, (time + t_r,RECOVER, id))


    def contact(self, time, id):

        friends = list(self.graph.neighbors(id))
        contacted_friend = random.choice(friends)

        # if self.graph.nodes[id]['state'] == 3:
        #     yield

        if self.graph.nodes[contacted_friend]['state'] == 0:

            print('#' + str(id) + ' has had contact with #{}.'.format(contacted_friend))
            u = random.uniform(0,1)

            if u < p_i:
                heapq.heappush(self.event_list, (time, INFECTION, contacted_friend))
        else:
            pass  # if in any other state than susceptible, this contact does not matter




        if self.graph.nodes[id]['state'] == 2:
            next_contact = (time+t_c, CONTACT, id)
            # if person is not infectious anymore, no need to schedule this
            heapq.heappush(self.event_list, next_contact)
        else:
            next_contact = False

        self.graph.nodes[id]['latest_contact'] = next_contact
        # this stores a pointer to the latest contact process of this id OR FALSE IF NONE IS SCHEDULED
        # it can be used to interrupt said process should the patient recover in the meantime



    def recover(self, time, id):

        # cancel related contact event
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



        self.update_state(id,3) # individuum is saved as recovered
        self.colormap[id] = 'grey'
        print(str(id)+' has recovered.')

        print('Contact process stopped due to recovery.')



# convenience:

    def sim(self, animation = True):
        # call first infection event
        event = (0, INFECTION, 0) # ind. #0 is infected at t = 0
        heapq.heappush(self.event_list, event)

        intervals = 1 # days for each animation frame
        counter = 0



        while self.event_list:
            event = heapq.heappop(self.event_list)
            self.do_event(event)

            current_t = event[0]
            if current_t >= counter * intervals:
                # ims.append(self.draw())
                import copy

                self.net_states.append((self.graph.copy(), self.colormap.copy()))
                counter += 1
        print('abc')
        pass




        print('simulation complete.')

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

    def last_sim_animated(self):
        assert self.net_states, "You need to run the simulation first!"


        fig = plt.figure()
        pos = self.pos
        def animate(idx):
            net_state = self.net_states[idx]

            graph = net_state[0]
            colormap = net_state[1]
            # fig.clf()

            nx.draw(graph, node_color = colormap, pos = pos)


        anim = animation.FuncAnimation(fig, animate, frames=len(self.net_states), interval=1000, blit=False)

        anim.save('test.mp4')
        print('saved animation.')

    # def animate(self,idx):
    #     if idx == 3:
    #         plt.scatter([1,2,3,4],[1,10,4,7])
    #         return
    #     pos = nx.spring_layout(self.graph, seed=100)
    #     nx.draw(self.net_states[idx], node_color = self.colormap, pos = pos)
    # def animate(self,i):
    #     colors = ['r', 'b', 'g', 'y', 'w', 'm']
    #     nx.draw_circular(self.graph, node_color=[random.choice(colors) for j in range(3)])

    # REMINDER:
# heapq uses lists as binary trees! So to go through it descendingly means the sequence
# 2k+2, 2k+1, 2(k-1)+2, ..., 1, 0 (i think)
# this is important for the canceling edge!




# from https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
# this is O(logn)
def heap_delete(h:list, i):
    h.pop()
    if i < len(h):
        heapq._siftup(h, i)
        heapq._siftdown(h, 0, i)



if __name__ == '__main__':
    net = Net(20,0.15,123456)
    # net.draw()

    net.sim()



    net.last_sim_animated()


