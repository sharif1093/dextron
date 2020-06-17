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
from dm_env import specs
from dm_control.suite import base
from dm_control.suite.utils import randomizers
from dm_control.utils import rewards

from dextron.zoo.hand_env.myquaternion import *


from dextron.zoo.common import get_model_and_assets_by_name

from .grasp_controllers import GraspControllerAllVelocity
from .grasp_controllers import GraspControllerIndividualVelocity
# from .grasp_controllers import GraspControllerAllPosition
# from .grasp_controllers import GraspControllerAllStep

from .trajectory_generator import SimulatedTrajectory
from .trajectory_generator import RealTrajectory
from .mocap_controller import MocapController

from PIL import Image
######################################################################################
## Model Constants and Tasks ##
###############################
_MODEL_NAME = "bb_left_hand"
## Teacher in reduced space. Robot in bigger space.
# _GRASP_CONTROLLER_AGENT = GraspControllerIndividualVelocity
# _GRASP_CONTROLLER_TEACHER = GraspControllerAllVelocity
## Robot and teacher be in the parameterized grasp:
_GRASP_CONTROLLER_AGENT = GraspControllerAllVelocity
_GRASP_CONTROLLER_TEACHER = GraspControllerAllVelocity
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
    
    def rgb2gray(self, rgb):
        # return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]), axis=0)
        return np.expand_dims(np.dot(rgb[...,:3], [.6, 0.1, 0.3]), axis=0)


    # def get_digit_contact_forces(self)

# Possible patterns:
#   * Height of torso with respect to foot: (self.named.data.xipos['torso', 'z'] - self.named.data.xipos['foot', 'z'])
#   * Horizontal speed of the Hopper:       self.named.data.sensordata['torso_subtreelinvel'][0]
#   * Signals from two foot touch sensors:  np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])


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
        # NOTE: We access to the "extra_env_kwargs" params not to the whole "explorer" params.
        self.params = params
        
        self.exclude_obs = self.params.get("exclude_obs", [])

        # Here we first try to ping "allow_demos". If it is absent we see if mode is "train". Then we allow training otherwise no.
        self.allow_demos = self.params.get("allow_demos", False)
        # pub_cameras: Whether to publish camera image or not.
        self.pub_cameras = self.params.get("pub_cameras", False)

        # All possible modes: train, test, eval, demo, replay
        self.explorer_mode = self.params.get("mode", "train")


        self.environment_kwargs = self.params.get("environment_kwargs")
        self.generator_args = self.params.get("generator_args")
        self.generator_type = self.params.get("generator_type")
        
        self.mode = None
        self.termination = None
        self.episode = 0
        self.counter = 0


        self.parameters = collections.OrderedDict()
        self.infos = collections.OrderedDict()

        super(Hand, self).__init__(random=self.params.get("random", None))




    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        Do the following steps:
        1. Randomize start/end positions.
        2. Produce trajectory at the beginning (or produce it stepwise if it takes a lot of time.)
        3. Assign the mocap pos: physics.named.data.mocap_pos["mocap"]
        4. Assign the mocap quat: physics.named.data.mocap_quat["mocap"]
        """
        # print(dir(physics.named.model))
        # print(physics.named.model.geom_quat)
        # exit()

        self.parameters = collections.OrderedDict()
        self.infos = collections.OrderedDict()
        self.termination = False
        self.episode += 1

        if self.allow_demos:
            # print(">>>> TEACHING in explorer [{}].".format(self.explorer_mode))
            self.mode = "teaching"
        else:
            # print(">>>> TRAINING in explorer [{}].".format(self.explorer_mode))
            self.mode = "training"
        

        # Grasp controllers in the teaching and training modes can be different.
        self.grasper = {}
        self.grasper["training"] = _GRASP_CONTROLLER_AGENT(physics)
        self.grasper["teaching"] = _GRASP_CONTROLLER_TEACHER(physics)


        
        if self.generator_type == "simulated":
            trajectory_class = SimulatedTrajectory
        elif self.generator_type == "real":
            # TODO: Randomize the time_limit but not the control_limit.
            # extracts_path = "/workspace/extracts"
            trajectory_class = RealTrajectory
        else:
            raise ValueError("'generator_type' is not valid.")
        
        trajectory = trajectory_class(self.environment_kwargs, self.generator_args)
        generator = trajectory.get_generator_list()
        self.mocap_controller = MocapController(generator)
        
        ######################################
        ## Initialization of mocap and hand ##
        ######################################
        # Move 1-step forward to set the initial pose of the mocap body.
        self.mocap_controller.step(physics)

        # NOTE: If you want to solve the issue of jumps in the hand position when setting the mocap position,
        #       you should also set the pos of the hand base to the corresponding adjacent value, so that they
        #       will be neighbors in the new config.
        ## Set the robot hand initial pose:
        offset = np.array([0,0.015,0], dtype=np.float32)
        if not self.mocap_controller.mocap_pos is None:
            physics.named.data.qpos["base_link_joint"][:3] = self.mocap_controller.mocap_pos + offset
        if not self.mocap_controller.mocap_quat is None:
            physics.named.data.qpos["base_link_joint"][3:7] = self.mocap_controller.mocap_quat
        

        ###################################
        ## Setting finger initial values ##
        ###################################
        # Start half-closed
        # t = 0.5 # np.random.rand()
        t = np.random.rand()
        physics.named.data.qpos["TPJ"] = -0.57*t #-0.57 0
        physics.named.data.qpos["IPJ"] =  1.57*t # 0 1.57
        physics.named.data.qpos["MPJ"] =  1.57*t # 0 1.57
        physics.named.data.qpos["RPJ"] =  1.57*t # 0 1.57
        physics.named.data.qpos["PPJ"] =  1.57*t # 0 1.57

        # Recording all random parameters
        self.parameters["initial_closure"] = np.array(t, dtype=np.float32)
        self.parameters["real_trajectory"] = np.array(self.generator_type == "real", dtype=np.uint8)
        ####
        # TODO: We may couple controller_gain and controller_thre. Because if threshold
        #       is smaller the controller should be faster.
        # self.parameters["controller_gain"] = 1.00 + np.random.rand() * (5 - 1)     # _GRASPER_GAIN = 2
        # self.parameters["controller_thre"] = 0.01 + np.random.rand() * (0.25-0.01) # _DISTANCE_NORMALIZED_CRITICAL = 0.15
        
        #  TODO: This is temporary. Ideally we have a very fast controller that closes just beside the hand:
        self.parameters["controller_gain"] = 1.00 + np.random.rand() * (10 - 1)     # _GRASPER_GAIN = 2
        # Seems to not have any effects!

        self.parameters["controller_thre"] = 0.01 + np.random.rand() * (0.25-0.01)
        # 0.15 # Should be in the range of 0.05 - 0.25

        # Store "str" data types in the info. Otherwise we will get errors.
        self.infos["rand"] = trajectory.parameters
        
        super(Hand, self).initialize_episode(physics)

        

    def action_spec(self, physics):
        """Returns a `BoundedArray` matching the `physics` actuators."""
        # TODO: "grasper" should be made here. Also, "grasper" should possibly determine the specs of agents.
        spec = collections.OrderedDict()
        # Setting actuator types, which is a BoundedArray:

        ## 1) Position/Velocity [continuous] control on individual actuators:
        # spec["agent"] = mujoco.action_spec(physics)
        ## 2) Velocity [discrete] control for all fingers simultaneously:
        # spec["agent"] = specs.BoundedArray(shape=(1,), dtype=np.int, minimum=0, maximum=_NUM_ACTION)
        ## 3) Position control for all fingers simultaneously:
        # spec["agent"] = specs.BoundedArray(shape=(1,), dtype=np.float, minimum=0, maximum=1) # 1: Fully closed || 0: Fully open
        ## 4) Velocity control (continuous) for all fingers simultaneously:
        spec["agent"]        = specs.BoundedArray(shape=(1,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing
        spec["demonstrator"] = specs.BoundedArray(shape=(1,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing
        ## 5) Velocity control (continuous) on individual fingers
        # TODO: Different action shapes MUST produce an error!
        ## spec["agent"]        = specs.BoundedArray(shape=(5,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing
        ## spec["demonstrator"] = specs.BoundedArray(shape=(1,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing

        # spec["agent"]        = specs.BoundedArray(shape=(5,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing
        # spec["demonstrator"] = specs.BoundedArray(shape=(5,), dtype=np.float, minimum=-1, maximum=1) # -1: Max speed openning || 1: Max speed closing

        
        # physics.model.nmocap
        # NOTE: In the current implementation we are not using "mocap" as an external action.
        #       "mocap" related actions are produced internally.
        #
        # spec["mocap"] = specs.BoundedArray(shape=(3,),
        #                                    dtype=np.float32,
        #                                    minimum=np.array([0, 0, 0 ]),
        #                                    maximum=np.array([.1,.1,.1]), # Are the maximum values correct?
        #                                    name="mocap")
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
            ctrl_actions = self.grasper["training"](action["agent"], physics)
        elif self.mode == "teaching":
            ctrl_actions = self.grasper["teaching"](action["demonstrator"], physics)
        
        # 3) Send low-level actions for execution in the simulator.
        super(Hand, self).before_step(ctrl_actions, physics)
        
        # Step the mocap body
        self.mocap_controller.step(physics)
        if self.mocap_controller.termination:
            self.termination = True
        




        # # TODO: Test the x and m:
        # #
        # # NOT PROPAGATED YET!
        # # x = physics.named.data.mocap_pos["mocap"].copy()
        # # q = physics.named.data.mocap_quat["mocap"].copy()
        # #
        # x = physics.named.data.xpos["mocap"].copy()
        # q = physics.named.data.xquat["mocap"].copy()
        # #
        # x_base = physics.named.data.xpos["base_link"].copy()
        # q_base = physics.named.data.xquat["base_link"].copy()
        # #
        # # x and orientation of palm_center.
        # xp = physics.named.data.site_xpos["palm_center"].copy()
        # mp = physics.named.data.site_xmat["palm_center"].copy()
        # #
        # # TODO: Goal here: Convert from "mocap" to "palm_center"; so we can do the same in our other file.
        # # TODO: Convert mp to qp. HOW?
        # # qw= √(1 + m00 + m11 + m22) /2
        # # qx = (m21 - m12)/( 4 *qw)
        # # qy = (m02 - m20)/( 4 *qw)
        # # qz = (m10 - m01)/( 4 *qw)
        # #
        # offset_local = np.array([0, 0.09, 0.03], dtype=np.float64)
        # # offset_rotated = quatTrans(quatConj(q), offset_local)
        # offset_rotated = quatTrans(quatConj(q_base), offset_local)
        # # print("++++++ Difference between palm_center and base:", np.linalg.norm(xp - (x_base + offset_rotated)))
        # #
        # #
        # offset_local_mocap = np.array([0, 0.09+0.015, 0.03], dtype=np.float64)
        # offset_local_mocap_rotated = quatTrans(quatConj(q), offset_local_mocap)
        # # print("++++++ Difference between palm_center and mocap:", np.linalg.norm(xp - (x + offset_local_mocap_rotated)))
        # # print("++++++ Quat diff between palm_center and base:", np.linalg.norm(q-q_base))
        # # print("")
        # #
        # # "xmat" format: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        # xmat = physics.named.data.site_xmat["palm_center"]
        # # print( np.linalg.norm(physics.named.data.xmat["mocap"] - physics.named.data.site_xmat["palm_center"]) )
        # #
        # #
        # normal_to_hand = quatTrans(quatConj(q), np.array([0,0,1]))
        # #
        # # This is what we wanted! The third row of the xmat matrix! Also, now we know how to obtain it from q!
        # # normal_to_hand    =====    [xmat[2],xmat[5],xmat[8]]
        # # 
        # print(normal_to_hand)
        # print(xmat[2],xmat[5],xmat[8])
        # print()
        # # 
        # #
        # # This is not what we want!!!!!!!!!!!!
        # # quatTrans(q, np.array([0,0,1]))     =====      xmat[6:9]
        # #
        # #
        # # print(normal_to_hand, xmat[6:9])
        # # 
        # # print("")
        # # 
        # # print("Z direction:", mp[:3])
        # #
        # # We want to know the direction of the b axis of the palm_center coordinate frame.
        # # b is the local z axis.
        # # Then, we want to make sure that we can obtain that axis from q.
        # # 
        # # print("palm_center matrix:", sum(mp[:3] * mp[3:6]), sum(mp[3:6] * mp[6:9]))
        # #
        # # xpos and size_xpos are in global coordinates. We should take care of this fact.
        # # 
        # # ----
        # # pos_offset_local  = np.array([0, 0, -0.025], dtype=np.float64)
        # # pos_offset_global = quatTrans(quatConj(q), pos_offset_local)
        # #
        # # Find the palm_center by coordinate transformations. Then check it versus the palm_center that we get from the following:
        



    def get_observation(self, physics):
        """Returns an observation of positions, velocities and touch sensors.
        We also include "info" in the observation. "info" must be a dictionary.

        Note:
            Nested dictionaries will not work with observations if used with flatten_observations.
            However, info can handle nested structures even with flatten_observations.
        
        Todo:
            Write a new flatten_observation function that can flatten nested observation dictionaries.
        """
        obs = collections.OrderedDict()

        #############################
        ### agent ###
        #############
        # Ignores horizontal position to maintain translational invariance:
        obs["agent"] = collections.OrderedDict()
        if not "position" in self.exclude_obs:
            obs["agent"]['position'] = physics.data.qpos[:].copy()
        if not "velocity" in self.exclude_obs:
            obs["agent"]['velocity'] = physics.data.qvel[:].copy()

        # obs['mocap_pos'] = physics.data.mocap_pos[:].copy()
        # obs['mocap_quat'] = physics.data.mocap_quat[:].copy()

        # obs['xpos_object'] = physics.named.data.xpos['long_cylinder'].copy()
        # obs['xquat_object'] = physics.named.data.xquat['long_cylinder'].copy()

        if not "rel_obj_hand" in self.exclude_obs:
            obs["agent"]["rel_obj_hand"] = physics.data.mocap_pos[:].copy() - physics.named.data.xpos['long_cylinder'].copy()
        if not "rel_obj_hand_dist" in self.exclude_obs:
            obs["agent"]["rel_obj_hand_dist"] = np.linalg.norm(physics.data.mocap_pos[:].copy() - physics.named.data.xpos['long_cylinder'].copy())
        if not "distance2" in self.exclude_obs:
            obs["agent"]["distance2"] = physics.get_distance_in_xy_plane()
        if not "closure" in self.exclude_obs:
            obs["agent"]["closure"] = physics.get_hand_closure()
            

        #############################
        ### Cameras ###
        ###############
        # render(height=240, width=320, camera_id=-1, overlays=(), depth=False, segmentation=False, scene_option=None)
        if self.pub_cameras:
            obs["camera"] = physics.rgb2gray(physics.render(width=60*4, height=60*3, camera_id="fixed")).astype(np.uint8)
            
            # image, depth = physics.render(width=60*4, height=60*3, camera_id="fixed", depth=True)
            # obs["camera"] = physics.rgb2gray(image).astype(np.uint8)
            # obs["depth"]  = depth

        # print(obs["camera"].shape)
        # (256, 4, 84, 84)


        # cam = physics.render(width=4*60*4, height=4*60*3, camera_id="fixed")
        # self.counter += 1
        # img = Image.fromarray(cam)
        # # img = img.convert("L")
        # img.save("/home/sharif/frames/{:04d}.jpg".format(self.counter))
        


        # 
        # print("position shape:", obs["agent"]['position'].shape)
        # print("velocity shape:", obs["agent"]['velocity'].shape)
        # print(physics.named.data.qpos)
        # print("--------------------------")
        # print(physics.named.data.qvel)

        # exit()
        



        
        # AGEN: (90, 4, 60, 80)
        # DEMO: (38, 4, 60, 80)
        # batch_size = 128

        # TODO: Velocity of the hand
        # TODO: Relative object/hand velocity

        # # obs['touch'] = physics.touch()
        
        #############################
        ### demonstrator ###
        ####################
        obs["demonstrator"] = collections.OrderedDict()
        obs['demonstrator']['distance'] = np.float32(physics.get_distance_in_xy_plane())
        obs['demonstrator']['hand_closure'] = np.float32(physics.get_hand_closure())

        #############################
        ### status ###
        ##############
        # We are putting these information in status, because we want it promptly in the observations.
        obs['status'] = collections.OrderedDict()
        is_training = 1 if self.mode == "training" else 0
        obs['status']['is_training'] = np.array(is_training, dtype=np.uint8)

        # The following also works for parameters:
        # obs['status']['parameters'] = self.parameters
        # obs['status']['time'] = np.array(physics.time())
        
        #############################
        ### parameters ###
        ##################
        obs['parameters'] = self.parameters

        #############################
        ### info ###
        ############
        # info has already been initialized in constructor and initial_episode.
        # obs['info'] = collections.OrderedDict()
        obs['info'] = self.infos
        # obs['info']['param1'] = np.array(1, dtype=np.float32)

        return obs

    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        height = physics.named.data.xipos['long_cylinder', 'z']


        # tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
        height_object = rewards.tolerance(height, bounds=(0.25, np.inf), margin=0)
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
        if self.termination or (height < 0.110):
            # `self.termination` is a class-level flag to indicate
            # the moment for termination.
            return 0.0
