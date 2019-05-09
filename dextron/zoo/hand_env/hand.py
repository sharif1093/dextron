"""Hand environment.
todo:
    Make the naive controller work with the cylinder object.
      * Make sure the distances are calculated properly (not considering z coordinate into account.)
      * Make sure the hand-closure is calculated properly.
      * Make sure the interface between GraspController/NaiveController is correct.
      * Calculate complex states in the upstream physics instance.
"""




# Are we in python2?
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sys

from dm_control import mujoco
from dm_control.rl import control
from dm_control.rl import specs
from dm_control.suite import base
from dm_control.suite.utils import randomizers
from dm_control.utils import rewards


from dextron.zoo.common import get_model_and_assets_by_name

from .grasp_controllers import GraspControllerAllVelocity
from .grasp_controllers import GraspControllerAllPosition
from .grasp_controllers import GraspControllerAllStep

from .naive_controller import NaiveController

######################################################################################
## Model Constants and Tasks ##
###############################
_MODEL_NAME = "bb_left_hand"
_GRASP_CONTROLLER = GraspControllerAllVelocity
# _NUM_ACTION = 2

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
    
    def get_distance_in_xy_plane(self):
        # We need the distance only in xy plane.
        # Only the first two elements.
        
        # mocap = self.named.data.mocap_pos["mocap"].copy()
        palm_center = self.named.data.site_xpos["palm_center"].copy()
        long_cylinder = self.named.data.xpos['long_cylinder'].copy()

        # dist = np.linalg.norm(mocap[0:2] - long_cylinder[0:2])
        dist = np.linalg.norm(palm_center[0:2] - long_cylinder[0:2])
        
        
        # self.data.mocap_pos[:].copy()
        return dist

    def get_joint_qpos(self, joint_name):
        return self.named.data.qpos[joint_name]

    # def get_joint_qpos_vec(self, joint_name_vec):
    #     joint_qpos_vec = np.zeros(shape=(len(joint_name_vec),))
    #     for index in range(len(joint_name_vec)):
    #         joint_qpos_vec[index] = self.get_joint_qpos(joint_name_vec[index])
    #     return joint_qpos_vec

    # def get_joint_specs(self, joint_name, joint_dir):
    #     joint_range = self.named.model.jnt_range[joint_name]
    #     joint_min = joint_range[0 if joint_dir>0 else 1] * joint_dir
    #     joint_max = joint_range[1 if joint_dir>0 else 0] * joint_dir
    #     return joint_min, joint_max

    def get_digit_closure(self, joint_name, joint_dir):
        # joint_min, joint_max = self.get_joint_specs(joint_name)
        joint_range = self.named.model.jnt_range[joint_name]
        joint_min = joint_range[0 if joint_dir>0 else 1] * joint_dir
        joint_max = joint_range[1 if joint_dir>0 else 0] * joint_dir

        joint_angle = self.get_joint_qpos(joint_name)
        
        joint_value = joint_angle * joint_dir
        # TODO: Because of saturation and different actuator ranges, the thumb
        #       moves faster than other fingers. So to compute hand_closure, we
        #       factor out all actuators of thumb altogether.
        joint_closure = (joint_value-joint_min) / (joint_max - joint_min)
        return joint_closure

    def get_hand_closure(self):
        # joint_names = ["TCJ", "TPJ", "IPJ", "MPJ", "RPJ", "PPJ"]
        joint_names = ["TPJ", "IPJ", "MPJ", "RPJ", "PPJ"]
        joint_dirs  = np.array([-1, 1, 1, 1, 1], dtype=np.float) # Can only be 1 or -1
        joint_closures = np.zeros(shape=(len(joint_names),))
        for index in range(len(joint_names)):
            joint_closures[index] = self.get_digit_closure(joint_names[index], joint_dirs[index])
        hand_closure = np.max(joint_closures)
        # hand_closure = np.mean(joint_closures)
        return hand_closure

    # def get_digit_contact_forces(self)

# Possible patterns:
#   * Height of torso with respect to foot: (self.named.data.xipos['torso', 'z'] - self.named.data.xipos['foot', 'z'])
#   * Horizontal speed of the Hopper:       self.named.data.sensordata['torso_subtreelinvel'][0]
#   * Signals from two foot touch sensors:  np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])

######################################################################################
#### Utility Functions ####
###########################

########################
# Trajectory Generator #
########################
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
######################################################################################



######################################################################################
#### Task ####
##############
class Hand(base.Task):
    """A Hand's `Task` to train "tasks"."""

    def __init__(self, **params):
        """Initialize an instance of `Hand`.

            random: Optional, either a `numpy.random.RandomState` instance, an
                integer seed for creating a new `RandomState`, or None to select a seed
                automatically (default).
        """
        self.params = params
        self.mode = None # "training" | "teaching"
        super(Hand, self).__init__(random=self.params.get("random", None))

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        Do the following steps:
        1. Randomize start/end positions.
        2. Produce trajectory at the beginning (or produce it stepwise if it takes a lot of time.)
        3. Assign the mocap pos: physics.named.data.mocap_pos["mocap"]
        4. Assign the mocap quat: physics.named.data.mocap_quat["mocap"]
        """

        self.grasper = _GRASP_CONTROLLER(physics)
        self.mode = "teaching"
        # self.mode = "training"
        
        #############################
        ### Mode-related settings ###
        #############################
        if self.mode == "teaching":
            self.teacher = NaiveController(self.action_spec(physics))
        elif self.mode == "training":
            self.teacher = None


        #########################################
        ### Randomizations of the environment ###
        #########################################
        a1 = np.random.rand()
        # For 10% of all times, start from a grasped position.
        offset = np.array([0,0.015,0], dtype=np.float32)
        if a1 > 1: # 0.8
            # # This part is not executed at all for now.
            # start_point = np.array([0.02, 0.00, 0.1], dtype=np.float32)

            # points = []
            # points.append(start_point)
            # points.append(np.array([0.02,  0.00,  0.3], dtype=np.float32))

            # times = [self.params["environment_kwargs"]["time_limit"]/3]
            pass

        else:
            # a0 = np.random.rand() * 0.01
            # a0 = 0 # No randomization at all!

            ## Randomizing initial position x coordinate on the x axis:
            # a0 = np.random.rand() * 0.04 - 0.1
            # start_point = np.array([a0, -0.35, 0.2], dtype=np.float32)

            ## Randomizing initial position x&y on a quadcircle:
            
            # NOTE: Don't start too close, give the agent some time.
            r = np.random.rand() * 0.25 + 0.1
            # r = np.random.rand() * 0.30 + 0.05
            # r = np.random.rand() * 0.35 + 0
            # r = 0.35
            # theta = np.random.rand() * np.pi / 6
            # theta = np.random.rand() * np.pi/3 - np.pi/6 + np.pi/7
            theta = np.random.rand() * np.pi/14 + np.pi/14
            # theta = np.pi / 7
            start_point = np.array([-r * np.sin(theta), -r * np.cos(theta), 0.1], dtype=np.float32)

            points = []
            points.append(start_point)
            points.append(np.array([0.02,  0.00,  0.1], dtype=np.float32))
            points.append(np.array([0.02,  0.00,  0.3], dtype=np.float32))
            
            times = [self.params["environment_kwargs"]["time_limit"]/2, self.params["environment_kwargs"]["time_limit"]/3]

        
        # NOTE: I noticed that qpos initializations do not always work as expected.
        #       Sometimes they just do not go the the given values.
        # Randomized initial finger positions by directly setting the joint values:
        # t = np.random.rand()
        t = 0.5
        physics.named.data.qpos["TPJ"] = -0.57*t #-0.57 0
        physics.named.data.qpos["IPJ"] = 1.57*t  # 0 1.57
        physics.named.data.qpos["MPJ"] = 1.57*t  # 0 1.57
        physics.named.data.qpos["RPJ"] = 1.57*t  # 0 1.57
        physics.named.data.qpos["PPJ"] = 1.57*t  # 0 1.57

        # Setting the mocap position:
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

        ## 1) Position [continuous] control on individual actuators:
        # spec["agent"] = mujoco.action_spec(physics)
        ## 2) Velocity [discrete] control for all fingers simultaneously:
        # spec["agent"] = specs.BoundedArraySpec(shape=(1,), dtype=np.int, minimum=0, maximum=_NUM_ACTION)
        ## 3) Position control for all fingers simultaneously:
        # spec["agent"] = specs.BoundedArraySpec(shape=(1,), dtype=np.float, minimum=0, maximum=1) # 1: Fully closed || 0: Fully open
        ## 4) Velocity control (continuous) for all fingers simultaneously:
        spec["agent"] = specs.BoundedArraySpec(shape=(1,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing

        
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
        
        if self.mode == "training":
            ctrl_actions = self.grasper(action["agent"], physics)
            super(Hand, self).before_step(ctrl_actions, physics)
        elif self.mode == "teaching":
            # 1) Take the teacher step
            actions = self.teacher.step(physics)
            # 2) Give the actions to the grasper for lower level translation.
            ctrl_actions = self.grasper(actions, physics)
            # 3) Send lowest-level actions for execution in the simulator.
            super(Hand, self).before_step(ctrl_actions, physics)
        
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
        # Ignores horizontal position to maintain translational invariance:
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()

        # obs['mocap_pos'] = physics.data.mocap_pos[:].copy()
        # obs['mocap_quat'] = physics.data.mocap_quat[:].copy()

        # obs['xpos_object'] = physics.named.data.xpos['long_cylinder'].copy()
        # obs['xquat_object'] = physics.named.data.xquat['long_cylinder'].copy()

        obs['rel_obj_hand'] = physics.data.mocap_pos[:].copy() - physics.named.data.xpos['long_cylinder'].copy()
        obs['rel_obj_hand_dist'] = np.linalg.norm(physics.data.mocap_pos[:].copy() - physics.named.data.xpos['long_cylinder'].copy())

        # TODO: Velocity of the hand
        # TODO: Relative object/hand velocity

        # obs['time'] = np.array(physics.time())

        # obs['touch'] = physics.touch()
        return obs

    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        height = physics.named.data.xipos['long_cylinder', 'z']


        # tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
        height_object = rewards.tolerance(height, bounds=(0.25,np.inf), margin=0)
        # height_object = (1 + height_object)/2

        reward = height_object
        
        # Commands of the agent to the robot in the current step:      physics.control()

        # TODO: With velocity-based controllers we can penalize the amount of actuation sent
        #       to actuators. We can penalize the sum over absolute values of finger actuations.

        # touch_data = np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])
        # if reward < 0:
        #     physics._reset_next_step = True
        #     # pass
        return reward
    
    def get_termination(self, physics):
        """Terminates when the we are not good."""
        height = physics.named.data.xipos['long_cylinder', 'z']
        if height < 0.120:
            return 0.0



