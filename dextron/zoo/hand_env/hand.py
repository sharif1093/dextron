"""Hopper domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite.utils import randomizers
from dm_control.utils import rewards
from dm_control.rl import specs
import numpy as np

from dextron.zoo.common import get_model_and_assets_by_name

import sys

#####################
## MODEL CONSTANTS ##
#####################
_MODEL_NAME = "bb_left_hand"
_NUM_ACTION = 2

# We want to delay actual running of this function until the last moment.
# Moreover, we can use this "get_model_and_assets" in different tasks if
# we have more than one.
get_model_and_assets = lambda: get_model_and_assets_by_name(_MODEL_NAME)


def grasp(**params): # environment_kwargs=None, 
    """Returns a Hand that strives."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hand(**params)
    environment_kwargs = params.get("environment_kwargs", {})
    return control.Environment(physics, task, **environment_kwargs)

######################################################################################
#### Extended Physics ####
##########################
# Add functions that can help your observation.
class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Hand domain."""
    pass
    # def object_height(self):

    # def mocapPos(self):
    #     return self.named
#     def height(self):
#         """Returns height of torso with respect to foot."""
#         return (self.named.data.xipos['torso', 'z'] -
#                 self.named.data.xipos['foot', 'z'])

#     def speed(self):
#         """Returns horizontal speed of the Hopper."""
#         return self.named.data.sensordata['torso_subtreelinvel'][0]

#     def touch(self):
#         """Returns the signals from two foot touch sensors."""
#         return np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])


######################################################################################
#### Trajectory Generator ####
##############################
def gen_traj_min_jerk(point_start, point_end, T, dt):
    t = 0
    R = 0.01*T
    
    N = point_start.shape[0]
    
    q = np.zeros((N,3))
    q[:,0] = point_start
    u = point_end
    while t<=T:
        D = T - t + R
        A = np.array([[0,1,0],[0,0,1],[-60/D**3,-36/D**2,-9/D]], dtype=np.float32)
        B = np.array([0,0,60/D**3], dtype=np.float32)
        
        for i in range(N):
            q[i,:] = q[i,:] + dt*(np.matmul(A,q[i,:])+B*u[i])
        
        t = t+dt
        #       pos   vel    acc
        yield q[:,0],q[:,1],q[:,2]


class Hand(base.Task):
    """A Hand's `Task` to train "tasks"."""

    def __init__(self, **params):
        """Initialize an instance of `Hand`.

            random: Optional, either a `numpy.random.RandomState` instance, an
                integer seed for creating a new `RandomState`, or None to select a seed
                automatically (default).
        """
        self.params = params
        super(Hand, self).__init__(random=self.params.get("random", None))

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        Do the following steps:
        1. Randomize start/end positions.
        2. Produce trajectory at the beginning (or produce it stepwise if it takes a lot of time.)
        3. Assign the mocap pos: physics.named.data.mocap_pos["mocap"]
        4. Assign the mocap quat: physics.named.data.mocap_quat["mocap"]
        """
        
        # print("--- Inertia of long_cylinder:", physics.named.model.body_inertia["long_cylinder"])
        # print("--- Mass of long_cylinder:", physics.named.model.body_mass["long_cylinder"])


        # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        # self._timeout_progress = 0

        # from pprint import pprint
        # pprint(dir(physics.model))
        ########## print("<<< ENVIRONMENT WAS RESET >>>")
        ### print("mocap_pos  for mocap:", physics.named.data.mocap_pos["mocap"])
        ### print("mocap_quat for mocap:", physics.named.data.mocap_quat["mocap"])
        # print("number of mocap bodies:", physics.model.nmocap)

        # Initialize the mocap location:
        # physics.named.data.mocap_pos["mocap"] = np.array([.1,.1,.1], dtype=np.float32) # .01*self.random.randn()
        
        
        ##############################
        ### Randomize a Trajectory ###
        ##############################
        a1 = np.random.rand()
        # For 10% of all times, start from a grasped position.
        offset = np.array([0,0.015,0], dtype=np.float32)
        if a1 > 0.8:
            start_point = np.array([0.01, 0.04, 0.2], dtype=np.float32)

            points = []
            points.append(start_point)
            points.append(np.array([0.01,  0.04,  0.3], dtype=np.float32))

            times = [self.params["environment_kwargs"]["time_limit"]/3]

        else:
            # a0 = np.random.rand() * 0.01
            # a0 = 0 # No randomization at all!

            ## Randomizing initial position x coordinate on the x axis:
            # a0 = np.random.rand() * 0.04 - 0.1
            # start_point = np.array([a0, -0.35, 0.2], dtype=np.float32)

            ## Randomizing initial position x&y on a quadcircle:
            theta = np.random.rand() * np.pi / 2
            start_point = np.array([-0.35*np.sin(theta), -0.35*np.cos(theta), 0.2], dtype=np.float32)

            points = []
            points.append(start_point)
            points.append(np.array([0.01,  0.04,  0.2], dtype=np.float32))
            points.append(np.array([0.01,  0.04,  0.3], dtype=np.float32))
            
            times = [self.params["environment_kwargs"]["time_limit"]/2, self.params["environment_kwargs"]["time_limit"]/3]

        physics.named.data.mocap_pos["mocap"] = start_point
        physics.named.data.xpos["base_link"] = start_point + offset

        self.mocap_traj = []
        for i in range(len(points)-1):
            self.mocap_traj.append(gen_traj_min_jerk(points[i], points[i+1], times[i],
                                   self.params["environment_kwargs"]["control_timestep"]))
        self.mocap_traj_index = 0

        # TODO: If you want to solve the issue of jumps in the hand position when setting the mocap position,
        #       you should also set the pos of the hand base to the corresponding adjacent value, so that they
        #       will be neighbors in the new config.
        

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        spec = collections.OrderedDict()
        # Setting actuator types, which is a BoundedArraySpec:
        spec["agent"] = mujoco.action_spec(physics)
        
        # physics.model.nmocap
        # NOTE: In the current implementation we are not using "mocap" as an external action.
        #       "mocap" related actions are produced internally.
        #
        # spec["mocap"] = specs.BoundedArraySpec(shape=(3,),
        #                                        dtype=np.float32,
        #                                        minimum=np.array([0, 0, 0 ]),
        #                                        maximum=np.array([.1,.1,.1]), # Are the maximum values correct?
        #                                        name="mocap")
        return spec

    def before_step(self, action, physics):
        """Updates the task from the provided action.
        
        Called by `control.Environment` before stepping the physics engine.

        Args:
        action: numpy array or array-like action values, or a nested structure of
            such arrays. Should conform to the specification returned by
            `self.action_spec(physics)`.
        physics: Instance of `Physics`.
        """
        ## Interpret the action
        # Setting the actuators to move. It calls physics.set_control
        # on the actuators to set their values:
        super(Hand, self).before_step(action["agent"], physics)
        # Setting the mocap bodies to move:
        
        try:
            mocap_pos, mocap_vel, mocap_acc = next(self.mocap_traj[self.mocap_traj_index])
            physics.named.data.mocap_pos["mocap"] = mocap_pos
        except StopIteration:
            if self.mocap_traj_index<len(self.mocap_traj)-1:
                self.mocap_traj_index += 1
        
        ###### physics.named.data.mocap_pos["mocap"] = action["mocap"]

        # We have a mocap_quat as well.
        



    def get_observation(self, physics):
        """Returns an observation of positions, velocities and touch sensors.
        We also include "info" in the observation. "info" must be a dictionary.
        """

        obs = collections.OrderedDict()
        # # Ignores horizontal position to maintain translational invariance:
        # obs['position'] = physics.data.qpos[1:].copy()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        # obs['position'] = physics.position()
        # obs['velocity'] = physics.velocity()

        # print("|position| =", len(obs['position']))
        # print("|velocity| =", len(obs['velocity']))

        # from pprint import pprint
        # pprint(dir(physics.data))
        # exit()
        obs['mocap_pos'] = physics.data.mocap_pos[:].copy()
        ##### obs['mocap_quat'] = physics.data.mocap_quat[:].copy()

        ## What about the object??
        # obs['mocap_pos'] = physics.data.mocap_pos[:].copy()
        # obs['mocap_quat'] = physics.data.mocap_quat[:].copy()

        obs['xpos_object'] = physics.named.data.xpos['long_cylinder'].copy()
        # obs['xquat_object'] = physics.named.data.xquat['long_cylinder'].copy()

        # print("shape of pos:", obs['position'].shape)

        obs['rel_obj_hand'] = obs['mocap_pos'] - obs['xpos_object']

        # print("type = ", physics.named.data.xpos['long_cylinder'])
        # exit()

        # if (obs['xpos'][0] != 0) and (obs['xpos'][1] != 0) and (obs['xpos'][2] != 0):
        #     print('xpos =', obs['xpos'])
        #     print('xquat =', obs['xquat'])


        ## obs['touch'] = physics.touch()
        return obs

    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        height = physics.named.data.xipos['long_cylinder', 'z']
        reward = 10 * (height-0.125)
        ##### print("reward = ", reward)

        # If the action is causing early termination, it should be penalized for that!
        # if self.get_termination(physics) == 0.0:
        #     reward = -10
    
        # from pprint import pprint
        # print(">>>>", physics.named.model.qpos0['long_cylinder']) 
        # print(physics.named.model.qpos0.item())

        # standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        # if self._hopping:
        #     hopping = rewards.tolerance(physics.speed(),
        #                                 bounds=(_HOP_SPEED, float('inf')),
        #                                 margin=_HOP_SPEED/2,
        #                                 value_at_margin=0.5,
        #                                 sigmoid='linear')
        #     return standing * hopping
        # else:
        #     small_control = rewards.tolerance(physics.control(),
        #                                       margin=1, value_at_margin=0,
        #                                       sigmoid='quadratic').mean()
        #     small_control = (small_control + 4) / 5
        #     return standing * small_control
        # if reward < 0:
        #     physics._reset_next_step = True
        #     # pass
        
        
        # if reward < -.1 or reward > 1:
        #     print("HERE WE ARE RESETTING ...")
        #     physics.reset()
        #     reward = 0
        return reward
    
    def get_termination(self, physics):
        """Terminates when the we are not good."""
        height = physics.named.data.xipos['long_cylinder', 'z']
        if height < 0.120:
            return 0.0



