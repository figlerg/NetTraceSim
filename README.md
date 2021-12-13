# Contact Tracing for Disease Containment: a Network-Based Analysis
NetTraceSim - ABM/DES modeling of COVID epidemic on social networks.

Author: Felix Gigler, Martin Bicher

## Introduction
The spread of SARS-CoV-2 is modeled by combining discrete events and an agent based approach, where the agents are seen as nodes in a social network.
Thus, we can investigate the impact of certain network properties on the epidemic.

## Installation
Install miniconda, then (in Anaconda Terminal):

	conda install numpy matplotlib networkx scipy ffmpeg

Alternatively, use the setup.py to create an venv.

Set 'kernels = <# of processor cores> -1' in do_experiment_parallel.py. This is system dependent.

## Use
The latest experiments can be found in 'Experiments/Paper'. 
Some of them might not work anymore due to changes in the base model. These could be adapted in the future.

Of particular interest to us are the following experiments:

3C_vary_C.py : Effect of TI and TTI for different amounts of clustering.

3E_vary_p_i.py : Effect of TI and TTI for different amounts of infectiousness.






