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


#################################
###           PRETEXT         ###
#################################
PUB_CAMERAS = False


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
cpanel["runner_name"]   = "dextron.pipeline.random_runner.RandomRunner"
cpanel["number_epochs"] = 15000  # epochs
cpanel["epoch_size"]    = 1000  # cycles
cpanel["test_activate"] = False # Test Activate
cpanel["test_interval"] = 10    # Test Interval Every #n Epochs
cpanel["test_win_size"] = 10    # Number of episodes to run test.
cpanel["save_interval"] = 10    # Save Interval Every #n Epochs


cpanel["scheduler_start"] = 0.3
cpanel["scheduler_steps"] = cpanel["epoch_size"] * 100
cpanel["scheduler_decay"] = 1.0 # Never reduce!
# cpanel["scheduler_decay"] = .95
# Using combined experience replay (CER) in the sampler.
cpanel["use_cer"] = False

cpanel["seed"] = 0
cpanel["cuda_deterministic"] = False # With TRUE we MIGHT get more deterministic results but at the cost of speed.

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
# cpanel["database_filename"] = "./workspace/parameters/for-input/session_20200706062600_blissful_mcnulty.csv"
cpanel["database_filename"] = None

cpanel["extracts_path"] = "./workspace/extracts"

cpanel["generator_type"] = "real" # "simulated" # "real"
cpanel["time_limit"] = 10.0  # Set the maximum time here!
cpanel["time_scale_offset"] = 0.5 # 1.0
cpanel["time_scale_factor"] = 2.5 # 2.0
cpanel["time_noise_factor"] = 0.8
cpanel["time_staying_more"] = 20  # timesteps
cpanel["reward_threshold"] = 1.0  # We are not interested in rewards < 5.0
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
    # Environment
    params["env"] = {}
    params["env"]["name"]   = cpanel["model_name"]
    
    params["env"]["from_module"] = cpanel.get("from_module", '')
    params["env"]["from_params"] = cpanel.get("from_params", False)

    if params["env"]["from_params"]:
        # For having environment from parameters
        from digideep.environment.dmc2gym.registration import EnvCreator
        from dextron.zoo.hand_env.hand import grasp

        task_kwargs = {"generator_type":cpanel["generator_type"], # Algorithm for generating trajectory: simulated/real
                       "generator_args":{"time_scale_offset":cpanel["time_scale_offset"],
                                         "time_scale_factor":cpanel["time_scale_factor"],
                                         "time_noise_factor":cpanel["time_noise_factor"],
                                         "time_staying_more":cpanel["time_staying_more"], # timesteps
                                         "extracts_path":cpanel["extracts_path"],
                                         "database_filename":cpanel["database_filename"]},
                       "random":None,
                       "pub_cameras":PUB_CAMERAS}
        
        # visualize_reward=True
        environment_kwargs = {"time_limit":cpanel["time_limit"], "control_timestep":cpanel["control_timestep"]}
        params["env"]["register_args"] = {"id":cpanel["model_name"],
                                          "entry_point":"digideep.environment.dmc2gym.wrapper:DmControlWrapper",
                                          "kwargs":{'dmcenv_creator':EnvCreator(grasp,
                                                                                task_kwargs=task_kwargs,
                                                                                environment_kwargs=environment_kwargs,
                                                                                visualize_reward=True),
                                                    'flat_observation':False,
                                                    'observation_key':"agent"}
                                         }

    ##############################################
    ### Normal Wrappers ###
    #######################
    norm_wrappers = []

    # Converting observation to 1 level
    # if not PUB_CAMERAS:
    #     norm_wrappers.append(dict(name="digideep.environment.wrappers.normal.WrapperLevelDictObs",
    #                             args={"path":cpanel["observation_key"],
    #                             },
    #                             enabled=True))

    # Normalizing actions (to be in [-1, 1])
    norm_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.WrapperNormalizeActDict",
                              args={"paths":["agent"]},
                              enabled=False))

    ##############################################
    ### Vector Wrappers ###
    #######################
    vect_wrappers = []
    
    # Normalizing rewards
    vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeRew",
                              args={"clip":5, # 10
                                    "gamma":cpanel["gamma"],
                                    "epsilon":1e-8
                              },
                              enabled=False)) # Not a good idea to normalize sparse rewards.
    

    # Log successful parameter sets for the expert policy
    vect_wrappers.append(dict(name="dextron.wrappers.success_logger.VecSuccessLogger",
                              request_for_args=["session"],
                              args={"threshold": cpanel["reward_threshold"], # Remove only zero-rewards
                                    "interval": 100, # How many episodes to print the log report?
                                    "num_workers": cpanel["num_workers"],
                                    "info_keys": ["/rand"],
                                    "obs_keys": ["/parameters"]
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

    #####################################
    # Runner: [episode < cycle < epoch] #
    #####################################
    params["runner"] = {}
    params["runner"]["name"] = cpanel.get("runner_name", "digideep.pipeline.Runner")
    params["runner"]["n_cycles"] = cpanel["epoch_size"]    # Meaning that 100 cycles are 1 epoch.
    params["runner"]["n_epochs"] = cpanel["number_epochs"] # Testing and savings are done after each epoch.
    params["runner"]["randargs"] = {'seed':cpanel["seed"], 'cuda_deterministic':cpanel["cuda_deterministic"]}
    params["runner"]["test_act"] = cpanel["test_activate"] # Test Activate
    params["runner"]["test_int"] = cpanel["test_interval"] # Test Interval
    params["runner"]["save_int"] = cpanel["save_interval"] # Save Interval

    
    params["agents"] = {}
    ##############################################
    ### Agent (#1) ### Demonstrator
    ##################
    params["agents"]["demonstrator"] = {}
    params["agents"]["demonstrator"]["name"] = "demonstrator"
    params["agents"]["demonstrator"]["type"] = "dextron.agent.demonstrator.NaiveController"
    params["agents"]["demonstrator"]["methodargs"] = {}
    agent_name = params["agents"]["demonstrator"]["name"]
    params["agents"]["demonstrator"]["methodargs"]["act_space"] = params["env"]["config"]["action_space"][agent_name]
    ##############################################



    # ##############################################
    # ### Memory ###
    # ##############
    params["memory"] = {}

    
    
    ##############################################
    ### Explorer ###
    ################
    params["explorer"] = {}

    params["explorer"]["train"] = {}
    params["explorer"]["train"]["mode"] = "train"
    params["explorer"]["train"]["env"] = params["env"]
    params["explorer"]["train"]["do_reset"] = False
    params["explorer"]["train"]["final_action"] = False
    params["explorer"]["train"]["warm_start"] = 0
    params["explorer"]["train"]["num_workers"] = cpanel["num_workers"]
    params["explorer"]["train"]["deterministic"] = False # MUST: Takes random actions
    params["explorer"]["train"]["n_steps"] = cpanel["n_steps"] # Number of steps to take a step in the environment
    params["explorer"]["train"]["n_episodes"] = None # Do not limit # of episodes
    params["explorer"]["train"]["win_size"] = 20 # Number of episodes to episode reward for report
    params["explorer"]["train"]["render"] = False
    params["explorer"]["train"]["render_delay"] = 0
    params["explorer"]["train"]["seed"] = cpanel["seed"] + 90
    params["explorer"]["train"]["extra_env_kwargs"] = {"mode":params["explorer"]["train"]["mode"], "allow_demos":False}

    params["explorer"]["test"] = {}
    params["explorer"]["test"]["mode"] = "test"
    params["explorer"]["test"]["env"] = params["env"]
    params["explorer"]["test"]["do_reset"] = True
    params["explorer"]["test"]["final_action"] = False
    params["explorer"]["test"]["warm_start"] = 0
    params["explorer"]["test"]["num_workers"] = cpanel["num_workers"] # We can use the same amount of workers for testing!
    params["explorer"]["test"]["deterministic"] = True   # MUST: Takes the best action
    params["explorer"]["test"]["n_steps"] = None # Do not limit # of steps
    params["explorer"]["test"]["n_episodes"] = cpanel["test_win_size"]
    params["explorer"]["test"]["win_size"] = cpanel["test_win_size"] # Extra episodes won't be counted
    params["explorer"]["test"]["render"] = False
    params["explorer"]["test"]["render_delay"] = 0
    params["explorer"]["test"]["seed"] = cpanel["seed"] + 100 # We want to make the seed of test environments different from training.
    params["explorer"]["test"]["extra_env_kwargs"] = {"mode":params["explorer"]["test"]["mode"], "allow_demos":False}

    params["explorer"]["eval"] = {}
    params["explorer"]["eval"]["mode"] = "eval"
    params["explorer"]["eval"]["env"] = params["env"]
    params["explorer"]["eval"]["do_reset"] = False
    params["explorer"]["eval"]["final_action"] = False
    params["explorer"]["eval"]["warm_start"] = 0
    params["explorer"]["eval"]["num_workers"] = 1
    params["explorer"]["eval"]["deterministic"] = True   # MUST: Takes the best action
    params["explorer"]["eval"]["n_steps"] = None # Do not limit # of steps
    params["explorer"]["eval"]["n_episodes"] = 1
    params["explorer"]["eval"]["win_size"] = -1
    params["explorer"]["eval"]["render"] = True
    params["explorer"]["eval"]["render_delay"] = 0
    params["explorer"]["eval"]["seed"] = cpanel["seed"] + 101 # We want to make the seed of eval environment different from test/train.
    params["explorer"]["eval"]["extra_env_kwargs"] = {"mode":params["explorer"]["eval"]["mode"], "allow_demos":cpanel.get("allow_demos", False)}
    ##############################################

    params["explorer"]["demo"] = {}
    params["explorer"]["demo"]["mode"] = "demo"
    params["explorer"]["demo"]["env"] = params["env"]
    params["explorer"]["demo"]["do_reset"] = False
    params["explorer"]["demo"]["final_action"] = False
    params["explorer"]["demo"]["warm_start"] = 0
    params["explorer"]["demo"]["num_workers"] = cpanel["num_workers"]
    params["explorer"]["demo"]["deterministic"] = False # MUST: Takes random actions
    params["explorer"]["demo"]["n_steps"] = cpanel["n_steps"] # Number of steps to take a step in the environment
    params["explorer"]["demo"]["n_episodes"] = None
    params["explorer"]["demo"]["win_size"] = -1
    params["explorer"]["demo"]["render"] = cpanel["render"]
    params["explorer"]["demo"]["render_delay"] = 0
    params["explorer"]["demo"]["seed"] = cpanel["seed"] + 50
    params["explorer"]["demo"]["extra_env_kwargs"] = {"mode":params["explorer"]["demo"]["mode"], "allow_demos":True}

    return params
