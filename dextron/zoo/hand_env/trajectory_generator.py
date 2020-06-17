import os, glob
import random
import json
import numpy as np
import numpy.matlib as npm

import quaternion
from scipy.interpolate import splprep, splev

from dextron.zoo.hand_env.myquaternion import *

######################################
### Trajectory Class ###
########################
class Trajectory:
    def __init__(self, environment_kwargs, generator_kwargs):
        self.environment_kwargs = environment_kwargs
        self.generator_kwargs = generator_kwargs
        self.parameters = {}
        self.initialize()

    def initialize(self):
        # Randomize all parameters relevant to the trajectory
        raise NotImplementedError()
        
    def generate(self):
        # yield
        raise NotImplementedError()

    def get_generator_list(self):
        raise NotImplementedError()




class RealTrajectory(Trajectory):
    # def __init__(self, environment_kwargs, generator_kwargs):
    #     Trajectory.__init__(self, environment_kwargs, generator_kwargs)
    
    # Utility function
    def read_trajectory_file(self, filename):
        t = []
        x = []
        q = []
        with open(filename, "r") as file:
            for line in file:
                entry = json.loads(line)
                t += [entry["t"]]
                x += [entry["x"]]
                q += [entry["q"]]
        
        t = np.array(t)
        x = np.array(x)
        q = np.array(q)

        # Offset time
        time_offset = t[0]
        t += -time_offset

        # Offset x
        x_offset = np.array([0.,0.,0.2])
        x += x_offset

        # Fix quaternion
        return t, x, q, time_offset
    
    def read_tag_from_metafile(self, metafile, tag):
        with open(metafile, "r") as file:
            annotations = json.load(file)
        
        for e in annotations:
            if e[0] == tag:
                return e[1]

        return None

    def quaternion_average(self, Q):
        # https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
        # Markley et al. (2007) Quaternion Averaging
        # Number of quaternions to average
        M = len(Q)
        A = npm.zeros(shape=(4,4))

        for i in range(0,M):
            q = Q[i]
            # multiply q with its transposed version q' and add A
            A = np.outer(q,q) + A

        # scale
        A = (1.0/M)*A
        # compute eigenvalues and -vectors
        eigenValues, eigenVectors = np.linalg.eig(A)
        # Sort by largest eigenvalue
        eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
        # return the real part of the largest eigenvector (has only real part)
        return np.real(eigenVectors[:,0].A1)

    # Override upper class functions
    def initialize(self):
        time_scale_offset = self.generator_kwargs["time_scale_offset"]
        time_scale_factor = self.generator_kwargs["time_scale_factor"]
        time_noise_factor = self.generator_kwargs["time_noise_factor"]
        extracts_path = self.generator_kwargs["extracts_path"]
        
        control_timestep  = self.environment_kwargs["control_timestep"]
        time_limit = self.environment_kwargs["time_limit"]

        #################################
        ### Sample a file name ###
        ##########################
        extracts_path_jsons = os.path.join(extracts_path, "*.json")
        all_samples = sorted(glob.glob(extracts_path_jsons))
        filename = random.choice(all_samples) # uniformly sampled
        # Based on the file name, we know from what region the trajectory starts.
        base_filename, _ = os.path.splitext(os.path.basename(filename))

        metafile = os.path.join(extracts_path, "meta", base_filename+"_meta.json")
        
        # Read the trajectory data from file
        t, x, q, time_offset = self.read_trajectory_file(filename)


        ########################################
        ### Process the trajectory ###
        ##############################
        # Randomize the trajectory duration (T)
        original_time = t[-1]
        noise = original_time + np.random.normal(scale=time_noise_factor)
        T = original_time * time_scale_factor + time_scale_offset + noise

        # Scale the t vector with respect to the trajectory time
        t_scaled = t * T / original_time
        t_vector = np.arange(0, T+1e-5, control_timestep)
        t_vector = t_vector[t_vector<=T] # Make sure that we do not go beyond the T.
        
        # 1. Fitting a spline curve on the x:
        try:
            x_spline, _ = splprep(x.T, u=t_scaled, s=0)
            x_vector = np.array([splev(t_vector[i], x_spline, ext=3) for i in range(len(t_vector))], dtype=np.float)
        except:
            print("Got an error with {} file while interpolating.".format(base_filename))
            exit()
            
        
        # 2. Fitting to quaternion arrays
        qs = quaternion.as_quat_array(q)
        q_vector = quaternion.as_float_array(quaternion.squad(qs, t_scaled, t_vector))
        
        # TODO: Make an interpolator in order to publish with custom timelimit/timestep.
        # Make a new T vector with the desired time-limit


        
        #################################################################
        ### Find and noise the position/orientation offset ###
        ######################################################
        # Finding averages of x and q when hand reaches the object.

        # Offset & Scaled reaching time
        reaching_time = (self.read_tag_from_metafile(metafile, tag="reached") - time_offset) * T / original_time

        # x_start = [x[i] for i in range(len(t)) if (t[i] < 0.1)]
        # Reaching State Duration
        RSD = 0.2
        x_reached = [x_vector[i] for i in range(len(t_vector)) if (t_vector[i] > reaching_time) and (t_vector[i] <= min(reaching_time + RSD, t_vector[-1]))]
        q_reached = [q_vector[i] for i in range(len(t_vector)) if (t_vector[i] > reaching_time) and (t_vector[i] <= min(reaching_time + RSD, t_vector[-1]))]

        x_grasp = sum(x_reached)/len(x_reached)
        q_grasp = self.quaternion_average(q_reached)
        
        # The results from SVD are almost the same as pure linear averaging.
        # q_grasp2 = -sum(q_reached)/len(q_reached)




        # Do for each element
        offset_local_mocap = np.array([0, 0.09+0.015, 0.03], dtype=np.float64)
        offset_local_mocap_rotated = quatTrans(quatConj(q_grasp), offset_local_mocap)
        x_palm_center_reached = x_grasp + offset_local_mocap_rotated
        x_palm_center_reached[2] = 0 # Zero out the z-coordinate

        # print("Palm Center norm should be close to 0:", np.linalg.norm(x_palm_center_reached))

        # Offset the x trajectory so end points are close to the "good" location for grasping.
        
        
        
        # TODO: This offset is something that needs to be randomized.
        # offset_noise_2d = np.array([-0.02, -0.01, 0], dtype = np.float64)
        offset_noise_x = -0.03 + np.random.rand() * (0.03 - (-0.03))
        offset_noise_y = -0.03 + np.random.rand() * (0.03 - (-0.03))
        offset_noise_2d = np.array([offset_noise_x, offset_noise_y, 0], dtype = np.float64)
        
        x_vector = x_vector - x_palm_center_reached + offset_noise_2d

        # TODO: What is the best measure for figuring out the orientation offset?
        # # Offset the end orientation
        # offset_support_point = np.array([0, 0.09+0.015, 0], dtype=np.float64)
        # offset_support_point_rotated = quatTrans(quatConj(q_grasp), offset_support_point)
        # x_palm_support_reached = x_grasp + offset_support_point_rotated

        # normal_to_hand = quatTrans(quatConj(q_grasp), np.array([0,0,1]))

        # px = normal_to_hand[0]
        # py = normal_to_hand[1]

        # p_vec = np.array([px, py], dtype=np.float64)
        # p_vec = p_vec / np.linalg.norm(p_vec)


        # # nx = -x_palm_center_reached[0]
        # # ny = -x_palm_center_reached[1]

        # nx = -x_palm_support_reached[0]
        # ny = -x_palm_support_reached[1]

        # n_vec = np.array([nx, ny], dtype=np.float64)
        # n_vec = n_vec / np.linalg.norm(n_vec)

        # print("Orientation vectors in 2D. Normal:", p_vec, ", From position:", n_vec)

        # # v_diff = np.array([px-nx, py-ny], dtype=np.float64)

        # # print("Orientation error should be close to 0:", np.linalg.norm(v_diff))
        # print()


        # How should "normal_to_hand" be?


        
        # TODO: Use x_grasp and q_grasp (which gives the hand pose when it tries grasping the object)
        #       to find a proper offset for the hand trajectory.
        
        # TODO: What is the acceptable pairs of (x,q) for grasping the object?
        
        # TODO: x is the position of some point attached to the user's wrist.
        #       Find the corresponding point in the robot's hand palm, which
        #       collides with the object center at the time of grasping.

        # TODO: 1. Find the position offset + salt it with some noise.
        #       2. Find the orientation offset + salt it with some noise.

        # TODO: Find the offset of the "palm point" from the mocap position.
        #       Then use the same kind of transformations used in "segmenter.py"
        #       to find the frame of that point in the palm.



        #################################
        ### Store trajectory data
        self.trajectory = {}
        self.trajectory["t_vector"] = t_vector
        self.trajectory["x_vector"] = x_vector
        self.trajectory["q_vector"] = q_vector
        #################################

        # print(f"Selected file: {base_filename} | Total time: {T} | dt: {control_timestep}")

        #################################
        ### Storing values
        self.parameters["filename"] = base_filename
        
        # Extract useful information from the sampled file name
        # subject_id, starting_flag, starting_id = base_filename.split("_")
        # self.parameters["subject_id"] = subject_id
        # self.parameters["starting_flag"] = starting_flag
        # self.parameters["starting_id"] = starting_id
        ##
        self.parameters["original_time"] = original_time
        self.parameters["randomized_time"] = T


        self.parameters["offset_noise_2d"] = offset_noise_2d
        #################################
    
    def generate(self):
        t_vector = self.trajectory["t_vector"]
        x_vector = self.trajectory["x_vector"]
        q_vector = self.trajectory["q_vector"]
        
        for i in range(len(t_vector)):
            ret_t = t_vector[i]
            ret_x = x_vector[i] # np.array(splev(ret_t, x_spline, ext=3))
            ret_q = q_vector[i]
            
            # For testing the neutral orientation of the mocap.
            # ret_q = np.array([1., 0., 0., 0.])
            
            yield ret_t, ret_x, ret_q
    
    def get_generator_list(self):
        return [self.generate()]



#####################################################################
### Simulated minimum-jerk trajectories ###
###########################################
class SimulatedTrajectory(Trajectory):
    def initialize(self):
        time_limit = self.environment_kwargs["time_limit"]
        
        ## Randomizing initial position x&y on a quadcircle:

        # theta = np.random.rand() * np.pi/7 + np.pi/7
        theta = np.random.rand() * np.pi/14 + np.pi/14
        
        # NOTE: Don't start too close, give the agent some time.
        r = np.random.rand() * 0.35 + 0.25
        start_point = np.array([-r * np.sin(theta), -r * np.cos(theta), 0.1], dtype=np.float32)
        
        ## PREVIOUSLY USED TO BE:
        ex = -0.045 # + 0.020 * (2*np.random.rand()-1)
        ey = -0.10

        # ex = -0.05  + 0.020 * (2*np.random.rand()-1)
        # ey = -0.095 + 0.020 * (2*np.random.rand()-1)

        ez = 0.10 + 0.020 * (2*np.random.rand()-1)

        approach_point = np.array([ex,  ey,  ez], dtype=np.float32)
        top_point = approach_point + np.array([0.00,  0.00,  0.2], dtype=np.float32)

        self.points = []
        self.points.append(start_point)
        self.points.append(approach_point)
        self.points.append(top_point)

        self.times = [time_limit/2, time_limit/3]


        #################################
        ### Storing values
        self.parameters["start_point"] = start_point
        self.parameters["approach_point"] = approach_point
        self.parameters["top_point"] = top_point
        #################################


    def generate(self, point_start, point_end, T, dt):
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
            #        pos   vel    acc
            #yield q[:,0],q[:,1],q[:,2]
            
            yield t, q[:,0], np.array([1,0,1,0],dtype=np.float32)
        

    def get_generator_list(self):
        control_timestep = self.environment_kwargs["control_timestep"]

        mocap_traj = []
        for i in range(len(self.points)-1):
            mocap_traj.append(self.generate(self.points[i], self.points[i+1], self.times[i], control_timestep))
        
        return mocap_traj
