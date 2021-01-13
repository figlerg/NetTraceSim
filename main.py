# code is in net.py and montecarlo.py

from net import Net
from montecarlo import monte_carlo
import pickle
from do_experiment import simple_experiment


# testing different network params




#experiments

mc_iterations = 30 # how many simulations per monte carlo
max_t = 100


# 1 baseline, on average people should have 100 friends here
n = 1000
p = 0.1
# if nothing happens, old objects are probably loaded from before! Look in Experiments.
net, counts = simple_experiment(n,p,mc_iterations, max_t)



# 2 same avg nr of friends, less individuals ->
n = 200
p = 0.5
# if nothing happens, old objects are probably loaded from before! Look in Experiments.
net, counts = simple_experiment(n,p,mc_iterations, max_t)



# 3 more friends
n = 1000
p = 0.5
# if nothing happens, old objects are probably loaded from before! Look in Experiments.
net, counts = simple_experiment(n,p,mc_iterations, max_t)


# 4 less friends
n = 1000
p = 0.02
# if nothing happens, old objects are probably loaded from before! Look in Experiments.
net, counts = simple_experiment(n,p,mc_iterations, max_t)

# 5 more individuals, more friends (than baseline)
n = 2000
p = 0.1
net, counts = simple_experiment(n,p,mc_iterations, max_t)
