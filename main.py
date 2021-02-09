# code is in net.py and montecarlo.py

from net import Net
from montecarlo import monte_carlo
import pickle
from do_experiment import simple_experiment


# testing different network params




# #experiments
#
# mc_iterations = 30 # how many simulations per monte carlo
# max_t = 200
#
#
# # 1 baseline, on average people should have 100 friends here
# n = 1000
# p = 0.1
# # if nothing happens, old objects are probably loaded from before! Look in Experiments.
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t)
#
#
#
# # 2 same avg nr of friends, less individuals ->
# n = 200
# p = 0.5
# # if nothing happens, old objects are probably loaded from before! Look in Experiments.
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t)
#
#
# n = 2000
# p = 0.05
# # if nothing happens, old objects are probably loaded from before! Look in Experiments.
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t)
#
#
#
# # 3 more friends
# n = 1000
# p = 0.5
# # if nothing happens, old objects are probably loaded from before! Look in Experiments.
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t)
#
#
# # 4 less friends
# n = 1000
# p = 0.005
# # if nothing happens, old objects are probably loaded from before! Look in Experiments.
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t)
#
# # 5 more individuals, more friends (than baseline)
# n = 2000
# p = 0.1
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t)
#
#
# # 6 same as 1, but with quarantine
# n = 1000
# p = 0.1
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, mode='quarantine')
#
# # 7 the next shall be a fast experiment to debug the tracing. simply
# mc_iterations = 10
# n = 200
# p = 0.5
#
# # # this is hardcoded for now
# # exp_7_path = r"C:\Users\Felix\Google Drive\Seminar_Networks\Experiments\comp_modes"
# #
# # mc_iterations = 20
# #
# # net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, force_recompute=True, path=exp_7_path)
# # net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, mode='quarantine', force_recompute=True,path=exp_7_path)
# # net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, mode='tracing', force_recompute=True,path=exp_7_path)
# # # net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(50,p,mc_iterations, max_t, mode='tracing', force_recompute=True,path=exp_7_path)
# # # net.animate_last_sim()
#
# # 8 testing some of the extreme cases

exp_8_path = r"C:\Users\giglerf\Google Drive\Seminar_Networks\Experiments\corner_cases"

mc_iterations = 3
max_t = 200

# n = 1000
# p = 0
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, force_recompute=False, path=exp_8_path)
#
# n = 1000
# p = 1
# net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, force_recompute=False, path=exp_8_path)

n = 1000
p = 1
net, counts, t_peak, peak_height, equilib_flag, durchseuchung = simple_experiment(n,p,mc_iterations, max_t, force_recompute=False, path=exp_8_path)

print([t_peak, peak_height, equilib_flag, durchseuchung])

