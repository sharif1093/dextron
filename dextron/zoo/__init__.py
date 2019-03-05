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

register(
    id="DMCHandGrasp-v0",
    entry_point="digideep.environment.dmc2gym.wrapper:DmControlWrapper",
    kwargs={'dmcenv_creator':EnvCreator(grasp, task_kwargs=None, environment_kwargs=None, visualize_reward=True),
            'flat_observation':True
           }
)


