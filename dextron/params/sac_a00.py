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
###          PREAMBLE         ###
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


# In case of multi-workers, change the following:
#   - cpanel["memory_size_in_chunks"] = int(5e3)
#   - cpanel["demo_memory_size_in_chunks"] = int(5e3)
#   - cpanel["num_workers"] = 20
#   - cpanel["n_update"] = 4


cpanel = OrderedDict()

#####################
### Runner Parameters
# num_frames = 10e6  # Number of frames to train
cpanel["runner_name"]   = "dextron.pipeline.replay_runner.PlaybackRunner"
cpanel["number_epochs"] = 1500  # epochs
cpanel["epoch_size"]    = 1000  # cycles
cpanel["test_activate"] = True  # Test activated
cpanel["test_interval"] = 10    # Test Interval Every #n Epochs
cpanel["test_win_size"] = 10    # Number of episodes to run test.
cpanel["save_interval"] = 50    # Save Interval Every #n Epochs
## Simulation will end when either time or max iterations exceed the following:
cpanel["max_exec_time"] = None   # hours
cpanel["max_exec_iter"] = None   # number of epochs


cpanel["scheduler_start"] = 0.3
cpanel["scheduler_steps"] = cpanel["epoch_size"] * 100
cpanel["scheduler_decay"] = 1.0 # Never reduce!
# cpanel["scheduler_decay"] = .95
# Using combined experience replay (CER) in the sampler.
cpanel["use_cer"] = False # This did not prove useful at all!

cpanel["seed"] = 0
cpanel["cuda_deterministic"] = False # With TRUE we MIGHT get more deterministic results but at the cost of speed.

#####################
### Memory Parameters
cpanel["keep_old_checkpoints"] = False
cpanel["memory_size_in_chunks"] = int(1e6)
cpanel["demo_memory_size_in_chunks"] = int(1e6)

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

cpanel["extracts_path"] = "./workspace/extracts"

cpanel["generator_type"] = "real" # "simulated" | "real"
# cpanel["generator_type"] = "simulated"
cpanel["time_limit"] = 10.0 # Set the maximum time here!
cpanel["time_scale_offset"] = 0.5 # 1.0
cpanel["time_scale_factor"] = 2.5 # 2.0
cpanel["time_noise_factor"] = 0.8
cpanel["time_staying_more"] = 20  # timesteps
cpanel["reward_threshold"] = 1.0  # We are not interested in rewards < 1.0
cpanel["control_timestep"] = 0.02 # "0.02" is a reasonable control_timestep. "0.04" is a reasonable fast-forward.
cpanel["exclude_obs"] = []

cpanel["gamma"] = 0.99     # The gamma parameter used in VecNormalize | Agent.preprocess | Agent.step

# # Wrappers
# cpanel["add_time_step"]          = False # It is suggested for MuJoCo environments. It adds time to the observation vector. CANNOT be used with renders.
# cpanel["add_image_transpose"]    = False # Necessary if training on Gym with renders, e.g. Atari games
# cpanel["add_dummy_multi_agent"]  = False # Necessary if the environment is not multi-agent (i.e. all dmc and gym environments),
#                                          # to make it compatibl with our multi-agent architecture.
# cpanel["add_vec_normalize"]      = True  # NOTE: USE WITH CARE. Might be used with MuJoCo environments. CANNOT be used with rendered observations.
# cpanel["add_frame_stack_axis"]   = False # Necessary for training on renders, e.g. Atari games. The nstack parameter is usually 4
#                                          # This stacks frames at a custom axis. If the ImageTranspose is activated
#                                          # then axis should be set to 0 for compatibility with PyTorch.

# cpanel["teaching_rate"] = 0.9

##################################
### Exploration/Exploitation Balance
### Exploration (~ num_workers * n_steps)
cpanel["num_workers"] = 1     # From Explorer           # Number of exploratory workers working together
cpanel["n_steps"] = 1         # From Explorer           # Number of frames to produce
cpanel["render"] = False # In the demo
### Exploitation (~ n_update * batch_size)
cpanel["n_update"] = 1        # From Agents: Updates per step
cpanel["batch_size"] = 32     # From Agents
cpanel["warm_start"] = 10000
cpanel["demo_use_ratio"] = 0.3 #  %rur of training data comes from the "replay", where (100-%rur) comes from the "train"
# cpanel["replay_use_ratio"] = 0.3 #  %rur of training data comes from the "replay", where (100-%rur) comes from the "train"
# cpanel["replay_nsteps"] = 1


#####################
### Agents Parameters
cpanel["agent_type"] = "digideep.agent.sac.Agent"
cpanel["lr_value"] = 3e-4
cpanel["lr_softq"] = 3e-4
cpanel["lr_actor"] = 3e-4

cpanel["hidden_size_value"] = 256
cpanel["hidden_size_softq"] = 256
cpanel["hidden_size_actor"] = 256

# cpanel["eps"] = 1e-5 # Epsilon parameter used in the optimizer(s) (ADAM/RMSProp/...)

# cpanel["noise_std"] = 0.2

cpanel["mean_lambda"] = 1e-3
cpanel["std_lambda"]  = 1e-3
cpanel["z_lambda"]    = 0.0

### Policy Parameters
cpanel["polyak_factor"] = 0.01
# cpanel["target_update_interval"] = 1

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
                       "pub_cameras":PUB_CAMERAS,
                       "exclude_obs":cpanel["exclude_obs"]}
        
        # visualize_reward=True
        environment_kwargs = {"time_limit":cpanel["time_limit"], "control_timestep":cpanel["control_timestep"]}
        params["env"]["register_args"] = {"id":cpanel["model_name"],
                                          "entry_point":"digideep.environment.dmc2gym.wrapper:DmControlWrapper",
                                          "kwargs":{'dmcenv_creator':EnvCreator(grasp,
                                                                                task_kwargs=task_kwargs,
                                                                                environment_kwargs=environment_kwargs,
                                                                                visualize_reward=False),
                                                    'flat_observation':False,
                                                    'observation_key':"agent"}
                                         }

    ##############################################
    ### Normal Wrappers ###
    #######################
    norm_wrappers = []

    # Converting observation to 1 level
    if not PUB_CAMERAS:
        norm_wrappers.append(dict(name="digideep.environment.wrappers.normal.WrapperLevelDictObs",
                                args={"path":cpanel["observation_key"],
                                },
                                enabled=True))
    # norm_wrappers.append(dict(name="digideep.environment.wrappers.normal.WrapperTransposeImage",
    #                           args={"path":"/camera"
    #                           },
    #                           enabled=True))
    # Normalizing actions (to be in [-1, 1])
    norm_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.WrapperNormalizeActDict",
                              args={"paths":["agent"]},
                              enabled=False))

    ##############################################
    ### Vector Wrappers ###
    #######################
    vect_wrappers = []

    if PUB_CAMERAS:
        vect_wrappers.append(dict(name="digideep.environment.wrappers.vector.VecFrameStackAxis",
                                args={"path":"/camera",
                                        "nstack":4, # By DQN Nature paper, it is called: phi length
                                        "axis":0},  # Axis=0 is required when ImageTransposeWrapper is called on the Atari games.
                                enabled=True))
    # Normalizing observations
    if not PUB_CAMERAS:
        vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeObsDict",
                                args={"paths":[cpanel["observation_key"]],
                                        "clip":10, # 5 or 10?
                                        "epsilon":1e-8
                                },
                                enabled=True))
    # Normalizing rewards
    vect_wrappers.append(dict(name="digideep.environment.wrappers.normalizers.VecNormalizeRew",
                              args={"clip":10, # 5 or 10?
                                    "gamma":cpanel["gamma"],
                                    "epsilon":1e-8
                              },
                              enabled=True)) # Not a good idea to normalize sparse rewards.
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
    params["runner"]["name"] = cpanel.get("runner_name", "digideep.pipeline.Runner")
    params["runner"]["max_time"] = cpanel.get("max_exec_time", None)
    params["runner"]["max_iter"] = cpanel.get("max_exec_iter", None)
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
    ### Agent (#1) ### Soft Actor-Critic
    ##################
    params["agents"]["agent"] = {}
    params["agents"]["agent"]["name"] = "agent"
    params["agents"]["agent"]["type"] = cpanel["agent_type"]
    params["agents"]["agent"]["observation_path"] = cpanel["observation_key"]
    params["agents"]["agent"]["methodargs"] = {}
    params["agents"]["agent"]["methodargs"]["n_update"] = cpanel["n_update"]  # Number of times to perform PPO update. Alternative name: PPO_EPOCH
    params["agents"]["agent"]["methodargs"]["gamma"] = cpanel["gamma"]  # Discount factor Gamma
    
    # params["agents"]["agent"]["methodargs"]["clamp_return"] = 1/(1-float(cpanel["gamma"]))
    # print("Clip Return =", params["agents"]["agent"]["methodargs"]["clamp_return"])

    params["agents"]["agent"]["methodargs"]["mean_lambda"] = cpanel["mean_lambda"]
    params["agents"]["agent"]["methodargs"]["std_lambda"] = cpanel["std_lambda"]
    params["agents"]["agent"]["methodargs"]["z_lambda"] = cpanel["z_lambda"]

    ################
    demo_batch_size = int(cpanel["demo_use_ratio"] * cpanel["batch_size"])
    train_batch_size  = cpanel["batch_size"] - demo_batch_size
    
    params["agents"]["agent"]["sampler_list"] = ["dextron.agent.sac.multi_sampler.multi_memory_sample"]
    params["agents"]["agent"]["sampler_args"] = {"agent_name":params["agents"]["agent"]["name"],
                                                 "batch_size":cpanel["batch_size"],
                                                 "scheduler_start":cpanel["scheduler_start"],
                                                 "scheduler_steps":cpanel["scheduler_steps"],
                                                 "scheduler_decay":cpanel["scheduler_decay"],
                                                 "batch_size_dict":{"train":train_batch_size, "demo":demo_batch_size},
                                                 "observation_path":params["agents"]["agent"]["observation_path"],
                                                 "use_cer":cpanel["use_cer"]
                                                }

    # # It deletes the last element from the chunk
    # params["agents"]["agent"]["sampler"]["truncate_datalists"] = {"n":1} # MUST be 1 to truncate last item: (T+1 --> T)

    #############
    ### Model ###
    #############
    agent_name = params["agents"]["agent"]["name"]
    observation_path = params["agents"]["agent"]["observation_path"]
    # params["agents"]["agent"]["policyname"] = "digideep.agent.sac.Policy"
    params["agents"]["agent"]["policyargs"] = {"obs_space": params["env"]["config"]["observation_space"][observation_path],
                                               "act_space": params["env"]["config"]["action_space"][agent_name],
                                               "image_repr_size": 80,
                                               "value_args": {"hidden_size": cpanel["hidden_size_value"], "init_w":0.003},
                                               "softq_args": {"hidden_size": cpanel["hidden_size_softq"], "init_w":0.003},
                                               "actor_args": {"hidden_size": cpanel["hidden_size_actor"], "init_w":0.003, "log_std_min":-20, "log_std_max":2},
                                               "average_args": {"mode":"soft", "polyak_factor":cpanel["polyak_factor"]},
                                                # # {"mode":"hard", "interval":10000}
                                               }
    
    # lim = params["env"]["config"]["action_space"][agent_name]["lim"][1][0]
    # # params["agents"]["agent"]["noisename"] = "digideep.agent.noises.EGreedyNoise"
    # # params["agents"]["agent"]["noiseargs"] = {"std":cpanel["noise_std"], "e":0.3, "lim": lim}
    
    # params["agents"]["agent"]["noisename"] = "digideep.agent.noises.OrnsteinUhlenbeckNoise"
    # params["agents"]["agent"]["noiseargs"] = {"mu":0, "theta":0.15, "sigma":cpanel["noise_std"], "lim":lim}
    # # params["agents"]["agent"]["noiseargs"] = {"mu":0, "theta":0.15, "sigma":1}

    params["agents"]["agent"]["optimname_value"] = "torch.optim.Adam"
    params["agents"]["agent"]["optimargs_value"] = {"lr":cpanel["lr_value"]}   # , "eps":cpanel["eps"]

    params["agents"]["agent"]["optimname_softq"] = "torch.optim.Adam"
    params["agents"]["agent"]["optimargs_softq"] = {"lr":cpanel["lr_softq"]}   # , "eps":cpanel["eps"]

    params["agents"]["agent"]["optimname_actor"] = "torch.optim.Adam"
    params["agents"]["agent"]["optimargs_actor"] = {"lr":cpanel["lr_actor"]}   # , "eps":cpanel["eps"]


    # RMSprop optimizer alpha
    # params["agents"]["agent"]["optimargs"] = {"lr":1e-2, "alpha":0.99, "eps":1e-5, "weight_decay":0, "momentum":0, "centered":False}
    ##############################################

    
    
    ##############################################
    ### Agent (#2) ### Demonstrator
    ##################
    params["agents"]["demonstrator"] = {}
    params["agents"]["demonstrator"]["name"] = "demonstrator"
    params["agents"]["demonstrator"]["type"] = "dextron.agent.demonstrator.NaiveController"
    params["agents"]["demonstrator"]["methodargs"] = {}
    agent_name = params["agents"]["demonstrator"]["name"]
    params["agents"]["demonstrator"]["methodargs"]["act_space"] = params["env"]["config"]["action_space"][agent_name]
    ##############################################



    ##############################################
    ### Memory ###
    ##############
    params["memory"] = {}

    # TODO: The memory size in chunks should be proportionately distributed. We think that "demo" should have a
    #       smaller memory size.

    # "digideep.memory.generic.Memory" | "digideep.memory.ringbuffer.Memory"
    # chunk_sample_len: Number of samples in a chunk
    # buffer_chunk_len: Number of chunks in the buffer

    params["memory"]["train"] = {}
    params["memory"]["train"]["type"] = "digideep.memory.ringbuffer.Memory"
    params["memory"]["train"]["args"] = {"name":"train",
                                         "keep_old_checkpoints":cpanel.get("keep_old_checkpoints", False),
                                         "chunk_sample_len":cpanel["n_steps"],
                                         "buffer_chunk_len":cpanel["memory_size_in_chunks"],
                                         "overrun":1}
    
    params["memory"]["demo"] = {}
    params["memory"]["demo"]["type"] = "digideep.memory.ringbuffer.Memory"
    params["memory"]["demo"]["args"] = {"name":"demo",
                                        "keep_old_checkpoints":cpanel.get("keep_old_checkpoints", False),
                                        "chunk_sample_len":cpanel["n_steps"],
                                        "buffer_chunk_len":cpanel["demo_memory_size_in_chunks"],
                                        "overrun":1}

    ##############################################

    
    
    ##############################################
    ### Explorer ###
    ################
    params["explorer"] = {}

    params["explorer"]["train"] = {}
    params["explorer"]["train"]["mode"] = "train"
    params["explorer"]["train"]["env"] = params["env"]
    params["explorer"]["train"]["do_reset"] = False
    params["explorer"]["train"]["final_action"] = False
    params["explorer"]["train"]["warm_start"] = cpanel["warm_start"] # In less than "warm_start" steps the agent will take random actions. 
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


#     params["explorer"]["replay"] = {}
#     params["explorer"]["replay"]["mode"] = "replay"
#     params["explorer"]["replay"]["env"] = params["env"]
#     params["explorer"]["replay"]["do_reset"] = False
#     params["explorer"]["replay"]["final_action"] = False
#     params["explorer"]["replay"]["warm_start"] = 0
#     params["explorer"]["replay"]["num_workers"] = cpanel["num_workers"]
#     params["explorer"]["replay"]["deterministic"] = False # MUST: Takes random actions
#     params["explorer"]["replay"]["n_steps"] = cpanel["replay_nsteps"] # Number of steps to take a step in the environment
#     params["explorer"]["replay"]["n_episodes"] = None
#     params["explorer"]["replay"]["win_size"] = 10
#     params["explorer"]["replay"]["render"] = False # False
#     params["explorer"]["replay"]["render_delay"] = 0
#     params["explorer"]["replay"]["seed"] = cpanel["seed"] + 50
#     params["explorer"]["replay"]["extra_env_kwargs"] = {"mode":params["explorer"]["replay"]["mode"], "allow_demos":False}

    return params
