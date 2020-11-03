"""
This parameter file is designed for continuous action environments.
For discrete action environments minor modifications might be required.

See Also:
    :ref:`ref-parameter-files`
"""

import numpy as np
from copy import deepcopy

from digideep.environment import MakeEnvironment
from collections import OrderedDict


# Memory address:
#     "/sessions/session_20200914233251_objective_mccarthy/memsnapshot/checkpoint-1000/demo.npz"



cpanel = OrderedDict()

#####################
### Runner Parameters
# num_frames = 10e6  # Number of frames to train
cpanel["runner_name"]   = "dextron.i_sv_trainer.Runner"
cpanel["number_epochs"] = 1000  # epochs
# cpanel["epoch_size"]    = 1000  # cycles
# cpanel["test_activate"] = False # Test Activate
# cpanel["test_interval"] = 10    # Test Interval Every #n Epochs
# cpanel["test_win_size"] = 10    # Number of episodes to run test.
# cpanel["save_interval"] = 100    # Save Interval Every #n Epochs
## Simulation will end when either time or max iterations exceed the following:

cpanel["max_exec_time"] = None   # hours
cpanel["max_exec_iter"] = None   # number of epochs


# cpanel["scheduler_start"] = 0.3
# cpanel["scheduler_steps"] = cpanel["epoch_size"] * 100
# cpanel["scheduler_decay"] = 1.0 # Never reduce!
# # cpanel["scheduler_decay"] = .95
# # Using combined experience replay (CER) in the sampler.
# cpanel["use_cer"] = False # This did not prove useful at all!

cpanel["seed"] = 0
cpanel["cuda_deterministic"] = False # With TRUE we MIGHT get more deterministic results but at the cost of speed.

#####################
### Memory Parameters
cpanel["keep_old_checkpoints"] = False
cpanel["demo_memory_size_in_chunks"] = cpanel["number_epochs"] * cpanel["epoch_size"]
# SHOULD be 1 for on-policy methods that do not have a replay buffer.
# SUGGESTIONS: 2^0 (~1e0) | 2^3 (~1e1) | 2^7 (~1e2) | 2^10 (~1e3) | 2^13 (~1e4) | 2^17 (1e5) | 2^20 (~1e6)

##########################
### Environment Parameters

cpanel["model_name"] = 'CustomDMCHandGrasp-v0'
if PUB_CAMERAS:
    cpanel["observation_key"] = "/camera"
else:
    cpanel["observation_key"] = "/agent"
cpanel["from_params"] = True

# Environment parameters
# cpanel["database_filename"] = "./workspace/parameters/session_20200622201351_youthful_pascal.csv"
cpanel["database_filename"] = "./workspace/parameters/for-input/session_20200706062600_blissful_mcnulty.csv"
# cpanel["database_filename"] = "./workspace/parameters/for-input/session_20200811120255_sharp_driscoll.csv"
# cpanel["database_filename"] = None

cpanel["extracts_path"] = "./workspace/extracts"

cpanel["generator_type"] = "real" # "simulated" | "real"
cpanel["time_limit"] = 10.0 # Set the maximum time here!
cpanel["time_scale_offset"] = 0.5 # 1.0
cpanel["time_scale_factor"] = 2.5 # 2.0
cpanel["time_noise_factor"] = 0.8
cpanel["time_staying_more"] = 20  # timesteps
cpanel["reward_threshold"] = 1.0  # We are not interested in rewards < 1.0
cpanel["control_timestep"] = 0.02 # "0.02" is a reasonable control_timestep. "0.04" is a reasonable fast-forward.

cpanel["gamma"] = 0.99     # The gamma parameter used in VecNormalize | Agent.preprocess | Agent.step

##################################
### Exploration/Exploitation Balance
### Exploration (~ num_workers * n_steps)
cpanel["num_workers"] = 1     # From Explorer           # Number of exploratory workers working together
cpanel["n_steps"] = 1         # From Explorer           # Number of frames to produce
cpanel["render"] = False

################################################################################
#########                      PARAMETER TREE                          #########
################################################################################
def gen_params(cpanel):
    params = {}

    #####################################
    # Runner: [episode < cycle < epoch] #
    #####################################
    params["runner"] = {}
    params["runner"]["name"] = cpanel["runner_name"]
    params["runner"]["max_time"] = cpanel.get("max_exec_time", None)
    params["runner"]["max_iter"] = cpanel.get("max_exec_iter", None)
    params["runner"]["n_epochs"] = cpanel["number_epochs"] # Testing and savings are done after each epoch.

    params["runner"]["randargs"] = {'seed':cpanel["seed"], 'cuda_deterministic':cpanel["cuda_deterministic"]}

    
    ##############################################
    ### Memory ###
    ##############
    params["memory"] = {}
    
    params["memory"]["demo"] = {}
    params["memory"]["demo"]["type"] = "digideep.memory.ringbuffer.Memory"
    params["memory"]["demo"]["args"] = {"name":"demo",
                                        "keep_old_checkpoints":cpanel.get("keep_old_checkpoints", False),
                                        "chunk_sample_len":cpanel["n_steps"],
                                        "buffer_chunk_len":cpanel["demo_memory_size_in_chunks"],
                                        "overrun":1}

    return params
