"""
See Also:
    :ref:`ref-parameter-files`
"""

import numpy as np
from copy import deepcopy

from digideep.environment import MakeEnvironment
from collections import OrderedDict

################################################################################
#########                       CONTROL PANEL                          #########
################################################################################
# The control panel brings the most important parameters to the top. It also
# helps to set all parameters that depends on a single value from one specific
# place:
#  - We can print and save this control panel instead of parameter list.
#  - The parameters here can also be taken from a YAML file.
#  - We can have default values now very easily.
#  - It provides semantic grouping of parameters
#  - We may unify the name of parameters which are basically the same in different
#    methods, but have different names.


cpanel = OrderedDict()

#####################
### Runner Parameters
# num_frames = 10e6  # Number of frames to train
cpanel["epoch_size"]    = 2  # cycles
cpanel["number_epochs"] = 100000
cpanel["test_activate"] = False # Test Activate
cpanel["test_interval"] = 10    # Test Interval Every #n Cycles
cpanel["save_interval"] = 1     # Save Interval Every #n Cycles

cpanel["seed"] = 0
cpanel["cuda_deterministic"] = False # With TRUE we MIGHT get more deterministic results but at the cost of speed.

#####################
### Memory Parameters
cpanel["memory_size_in_chunks"] = int(1) # SHOULD be 1 for on-policy methods that do not have a replay buffer.
# SUGGESTIONS: 2^0 (~1e0) | 2^3 (~1e1) | 2^7 (~1e2) | 2^10 (~1e3) | 2^13 (~1e4) | 2^17 (1e5) | 2^20 (~1e6)

##########################
### Environment Parameters
# cpanel["model_name"] = 'DMCHandGrasp-v0'
# cpanel["from_module"] = 'dextron.zoo'

cpanel["model_name"] = 'CustomDMCHandGrasp-v0'
cpanel["from_params"] = True
# Environment parameters
cpanel["time_limit"] = 6

cpanel["observation_key"] = "/agent"
cpanel["gamma"] = 0.99     # The gamma parameter used in VecNormalize | Agent.preprocess | Agent.step

# # Wrappers
# cpanel["add_time_step"]          = False # It is suggested for MuJoCo environments. It adds time to the observation vector. CANNOT be used with renders.
# cpanel["add_image_transpose"]    = False # Necessary if training on Gym with renders, e.g. Atari games
# cpanel["add_dummy_multi_agent"]  = True  # Necessary if the environment is not multi-agent (i.e. all dmc and gym environments),
#                                          # to make it compatibl with our multi-agent architecture.
# cpanel["add_vec_normalize"]      = False # NOTE: USE WITH CARE. Might be used with MuJoCo environments. CANNOT be used with rendered observations.
# cpanel["add_frame_stack_axis"]   = False # Necessary for training on renders, e.g. Atari games. The nstack parameter is usually 4
#                                          # This stacks frames at a custom axis. If the ImageTranspose is activated
#                                          # then axis should be set to 0 for compatibility with PyTorch.
# # TODO: Action normalizer and clipper


##################################
### Exploration/Exploitation Balance
### Exploration (~ num_workers * n_steps)
cpanel["num_workers"] = 1  # From Explorer           # Number of exploratory workers working together
cpanel["n_steps"] = 256    # From Explorer           # Number of frames to produce
### Exploitation (~ n_update * batch_size): [PPO_EPOCH] Number of times to perform PPO update, i.e. number of frames to process.
cpanel["n_update"] = 4     # From Agents
cpanel["batch_size"] = 128 # From Agents

cpanel["num_mini_batches"] = 8



#####################
### Agents Parameters
cpanel["use_gae"] = True   # Whether to use GAE to calculate returns or not.
cpanel["tau"] = 0.95       # The parameter used for calculating advantage function.
cpanel["recurrent"] = False
cpanel["actor_feature_size"] = 64

cpanel["lr"] = 3e-4 # 2.5e-4 | 7e-4
cpanel["eps"] = 1e-5 # Epsilon parameter used in the optimizer(s) (ADAM/RMSProp/...)

cpanel["clip_param"] = 0.1       # 0.2  # PPO clip parameter
cpanel["value_loss_coef"] = 0.50 # 1    # Value loss coefficient
cpanel["entropy_coef"] = 0.00    # 0.01 # Entropy term coefficient
cpanel["max_grad_norm"] = 0.50   # Max norm of gradients
cpanel["use_clipped_value_loss"] = True

# add_vec_normalize

################################################################################
#########                      PARAMETER TREE                          #########
################################################################################
def gen_params(cpanel):
    params = {}
    # Environment
    params["env"] = {}
    params["env"]["name"] = cpanel["model_name"]
    
    params["env"]["from_module"] = cpanel.get("from_module", '')
    params["env"]["from_params"] = cpanel.get("from_params", False)
    
    if params["env"]["from_params"]:
        # For having environment from parameters
        from digideep.environment.dmc2gym.registration import EnvCreator
        from dextron.zoo.hand_env.hand import grasp

        task_kwargs = {"random":None}
        environment_kwargs = {"time_limit":cpanel["time_limit"], "control_timestep":0.02}
        params["env"]["register_args"] = {"id":cpanel["model_name"],
                                          "entry_point":"digideep.environment.dmc2gym.wrapper:DmControlWrapper",
                                          "kwargs":{'dmcenv_creator':EnvCreator(grasp,
                                                                                task_kwargs=task_kwargs,
                                                                                environment_kwargs=environment_kwargs,
                                                                                visualize_reward=True),
                                                    'flat_observation':True}
                                         }
    ##########

    
    ##############################################
    ### Normal Wrappers ###
    #######################
    norm_wrappers = []
    
    norm_wrappers.append(dict(name="digideep.environment.wrappers.normal.WrapperLevelDictObs",
                              args={"path":cpanel["observation_key"],
                              },
                              enabled=False))
    norm_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.WrapperNormalizeActDict",
                              args={"paths":["agent"]},
                              enabled=False))
    ##############################################
    ### Vector Wrappers ###
    #######################
    vect_wrappers = []

    vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeObsDict",
                              args={"paths":[cpanel["observation_key"]],
                                    "clip":5, # 10
                                    "epsilon":1e-8
                              },
                              enabled=True))
    vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeRew",
                              args={"clip":5, # 10
                                    "gamma":cpanel["gamma"],
                                    "epsilon":1e-8
                              },
                              enabled=True))
    ##############################################
    params["env"]["main_wrappers"] = {"Monitor":{"allow_early_resets":True, # We need it to allow early resets in the test environment.
                                                 "reset_keywords":(),
                                                 "info_keywords":()},
                                      "WrapperDummyMultiAgent":{"agent_name":"agent"},
                                      "WrapperDummyDictObs":{"observation_key":"agent"}
                                     }
    params["env"]["norm_wrappers"] = norm_wrappers
    params["env"]["vect_wrappers"] = vect_wrappers


    menv = MakeEnvironment(session=None, mode=None, seed=1, **params["env"])
    params["env"]["config"] = menv.get_config()

    # Some parameters
    # params["env"]["gamma"] = 1-1/params["env"]["config"]["max_steps"] # 0.98



    #####################################
    # Runner: [episode < cycle < epoch] #
    #####################################
    params["runner"] = {}
    params["runner"]["n_cycles"] = cpanel["epoch_size"]    # Meaning that 100 cycles are 1 epoch.
    params["runner"]["n_epochs"] = cpanel["number_epochs"] # Testing and savings are done after each epoch.
    params["runner"]["randargs"] = {'seed':cpanel["seed"], 'cuda_deterministic':cpanel["cuda_deterministic"]}
    params["runner"]["test_act"] = cpanel["test_activate"] # Test Activate
    params["runner"]["test_int"] = cpanel["test_interval"] # Test Interval
    params["runner"]["save_int"] = cpanel["save_interval"] # Save Interval

    # We "save" after each epoch is done.
    # We "test" after each epoch is done.


    
    params["agents"] = {}
    ##############################################
    ### Agent (#1) ###
    ##################
    params["agents"]["agent"] = {}
    params["agents"]["agent"]["name"] = "agent"
    params["agents"]["agent"]["type"] = "digideep.agent.PPO"
    params["agents"]["agent"]["observation_path"] = cpanel["observation_key"]
    params["agents"]["agent"]["methodargs"] = {}
    params["agents"]["agent"]["methodargs"]["n_steps"] = cpanel["n_steps"]  # Same as "num_steps" / T
    params["agents"]["agent"]["methodargs"]["n_update"] = cpanel["n_update"]  # Number of times to perform PPO update. Alternative name: PPO_EPOCH
    params["agents"]["agent"]["methodargs"]["clip_param"] = cpanel["clip_param"]  # PPO clip parameter
    params["agents"]["agent"]["methodargs"]["value_loss_coef"] = cpanel["value_loss_coef"]  # Value loss coefficient
    params["agents"]["agent"]["methodargs"]["entropy_coef"] = cpanel["entropy_coef"]  # Entropy term coefficient
    params["agents"]["agent"]["methodargs"]["max_grad_norm"] = cpanel["max_grad_norm"]  # Max norm of gradients
    params["agents"]["agent"]["methodargs"]["use_clipped_value_loss"] = cpanel["use_clipped_value_loss"]

    
    params["agents"]["agent"]["sampler"] = {}
    params["agents"]["agent"]["sampler"]["agent_name"] = params["agents"]["agent"]["name"]
    params["agents"]["agent"]["sampler"]["num_mini_batches"] = cpanel["num_mini_batches"]
    params["agents"]["agent"]["sampler"]["compute_advantages"] = {"gamma":cpanel["gamma"], # Discount factor for rewards
                                                                  "tau":cpanel["tau"], # GAE parameter
                                                                  "use_gae":cpanel["use_gae"]}
    # It deletes the last element from the chunk
    params["agents"]["agent"]["sampler"]["truncate_datalists"] = {"n":1} # MUST be 1 to truncate last item: (T+1 --> T)
    params["agents"]["agent"]["sampler"]["observation_path"] = params["agents"]["agent"]["observation_path"]

    #############
    ### Model ###
    #############
    agent_name = params["agents"]["agent"]["name"]
    observation_path = params["agents"]["agent"]["observation_path"]
    params["agents"]["agent"]["policyname"] = "digideep.agent.policy.stochastic.Policy"
    params["agents"]["agent"]["policyargs"] = {"obs_space": params["env"]["config"]["observation_space"][observation_path],
                                               "act_space": params["env"]["config"]["action_space"][agent_name],
                                               "modelname": "digideep.model.models.MLPModel",
                                               "modelargs": {"recurrent":cpanel["recurrent"], "output_size":cpanel["actor_feature_size"]}
                                               }
    params["agents"]["agent"]["optimname"] = "torch.optim.Adam"
    params["agents"]["agent"]["optimargs"] = {"lr":cpanel["lr"], "eps":cpanel["eps"]}

    # RMSprop optimizer apha
    # params["agents"]["agent"]["optimargs"] = {"lr":1e-2, "alpha":0.99, "eps":1e-5, "weight_decay":0, "momentum":0, "centered":False}
    ##############################################


    ##############################################
    ### Memory ###
    ##############
    params["memory"] = {}

    # Number of samples in a chunk
    params["memory"]["chunk_sample_len"] = cpanel["n_steps"]
    # Number of chunks in the buffer:
    params["memory"]["buffer_chunk_len"] = cpanel["memory_size_in_chunks"]
    ##############################################

    
    ##############################################
    ### Explorer ###
    ################
    params["explorer"] = {}

    params["explorer"]["train"] = {}
    params["explorer"]["train"]["mode"] = "train"
    params["explorer"]["train"]["env"] = params["env"]
    params["explorer"]["train"]["do_reset"] = False
    params["explorer"]["train"]["final_action"] = True
    params["explorer"]["train"]["num_workers"] = cpanel["num_workers"]
    params["explorer"]["train"]["deterministic"] = False # MUST: Takes random actions
    params["explorer"]["train"]["n_steps"] = cpanel["n_steps"] # Number of steps to take a step in the environment
    params["explorer"]["train"]["render"] = False
    params["explorer"]["train"]["render_delay"] = 0
    params["explorer"]["train"]["seed"] = cpanel["seed"] # + 3500

    params["explorer"]["test"] = {}
    params["explorer"]["test"]["mode"] = "test"
    params["explorer"]["test"]["env"] = params["env"]
    params["explorer"]["test"]["do_reset"] = False
    params["explorer"]["test"]["final_action"] = False
    params["explorer"]["test"]["num_workers"] = 1 # cpanel["num_workers"] # We can use the same amount of workers for testing!
    params["explorer"]["test"]["deterministic"] = True   # MUST: Takes the best action
    params["explorer"]["test"]["n_steps"] = params["env"]["config"]["max_episode_steps"] # Number of steps to take a step in the environment
    params["explorer"]["test"]["render"] = False
    params["explorer"]["test"]["render_delay"] = 0
    params["explorer"]["test"]["seed"] = cpanel["seed"] + 100 # We want to make the seed of test environments different from training.

    params["explorer"]["eval"] = {}
    params["explorer"]["eval"]["mode"] = "eval"
    params["explorer"]["eval"]["env"] = params["env"]
    params["explorer"]["eval"]["do_reset"] = False
    params["explorer"]["eval"]["final_action"] = False
    params["explorer"]["eval"]["num_workers"] = 1
    params["explorer"]["eval"]["deterministic"] = True   # MUST: Takes the best action
    params["explorer"]["eval"]["n_steps"] = params["env"]["config"]["max_episode_steps"] # Number of steps to take a step in the environment
    params["explorer"]["eval"]["render"] = True
    params["explorer"]["eval"]["render_delay"] = 0
    params["explorer"]["eval"]["seed"] = cpanel["seed"] + 101 # We want to make the seed of eval environment different from test/train.
    ##############################################

    return params
