# this file includes the class for individual agents using the simpy architecture
import simpy as sp

colormap = {0:'white', 1:'yellow', 2:'red', 3:'green'}
strmap = {0:'susceptible', 1:'exposed', 2:'infectious', 3:'recovered'}

class Person(object):

    def __init__(self, id:int, state:int):
        self.state = state
        assert self.state in [0,1,2,3]  # susc, exp, inf, rec in this order are the only states supported

        self.id = id

    def infection(self, t):
        # TODO enable counting for overall populus?

        self.state = 1
        print('Person #{} was infected on day {}.'.format(self.id, t))


#convenience functions
    def color(self):
        return colormap(self.state)

    def __str__(self):
        return "Person #{} is {}.".format(str(self.id), strmap[self.state])
