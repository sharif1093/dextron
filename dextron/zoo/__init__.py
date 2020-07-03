"""This is the entrypoint to the dm_control package.
None of the other files/modules are allowed to call
any module/submodule of the dm_control package.
At here, we first call the hax related to the dm_control,
then we continue.

NOTE: Go to dm_control/render/glfw_renderer.py, move the
      `glfw.init()` part at the beginning of the `_platform_init`
      function.
"""
# This line MUST be the first line
# import digideep.environment.dmc2gym


from digideep.environment.dmc2gym.registration import EnvCreator
from dextron.zoo.hand_env.hand import grasp
from gym.envs.registration import register

# print(">>>>>>>>>>>>>>>>> zoo.__init__ is now LOADED")

_CONTROL_TIMESTEP = .02 # (Seconds)
_DEFAULT_TIME_LIMIT = 6 # Default duration of an episode, in seconds.
_DEFAULT_TIME_SCALE_OFFSET = 0.5 # 1.0
_DEFAULT_TIME_SCALE_FACTOR = 2.5 # 2.0
_DEFAULT_TIME_NOISE_FACTOR = 0.8

task_kwargs = {"generator_type":"real",
               "generator_args":{"time_scale_offset":0.5, #1.0
                                 "time_scale_factor":2.5, #2.0
                                 "time_noise_factor":0.8, #
                                 "time_staying_more":20,  # timesteps
                                 "extracts_path":"/workspace/extracts",
                                 "database_filename":None},
               "random":None,}

environment_kwargs = {"time_limit":_DEFAULT_TIME_LIMIT, "control_timestep":_CONTROL_TIMESTEP}

register(
    id="DMCHandGrasp-v0",
    entry_point="digideep.environment.dmc2gym.wrapper:DmControlWrapper",
    kwargs={'dmcenv_creator':EnvCreator(grasp, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs, visualize_reward=True),
            'flat_observation':False,
            'observation_key':"agent"
           }
)


