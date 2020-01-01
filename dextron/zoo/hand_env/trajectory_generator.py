import os, glob
import random
import json
import numpy as np

import quaternion
from scipy.interpolate import splprep, splev
#####################################################################
### Sampled real trajectories ###
#################################

# Read the file
def read_trajectory_file(filename):
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
    t += -t[0]

    # Offset x
    x_offset = np.array([0.,0.,0.2])
    x += x_offset

    # Fix quaternion

    return t, x, q

def gen_traj_real(time_limit, control_timestep, time_scale_offset, time_scale_factor, time_noise_factor, extracts_path=".."):
    """
    control_timestep: dt
    """
    # Sample a file name
    extracts_path = os.path.join(extracts_path, "*.json")
    all_samples = sorted(glob.glob(extracts_path))

    uniformly_sampled_trajectory_file = random.choice(all_samples)
    # Based on the file name, we know from what region the trajectory starts.
    base_filename, _ = os.path.splitext(os.path.basename(uniformly_sampled_trajectory_file))
    subject_id, starting_flag, starting_id = base_filename.split("_")

    # Read the file
    t, x, q = read_trajectory_file(uniformly_sampled_trajectory_file)

    # Randomize the trajectory duration (T)
    natural_time = t[-1]
    noise = natural_time + np.random.normal(scale=time_noise_factor)
    T = natural_time * time_scale_factor + time_scale_offset + noise

    print("Selected file: {} | Total time: {} | dt: {}".format(base_filename, T, control_timestep))

    t_scaled = t * T / natural_time
    t_vector = np.arange(0, T+1e-5, control_timestep)
    t_vector = t_vector[t_vector<=T] # Make sure that we do not go beyond the T.
    
    # 1. Fitting a spline curve on the x:
    tck, _ = splprep(x.T, u=t_scaled, s=0)
    
    # 2. Fitting to quaternion arrays
    qs = quaternion.as_quat_array(q)
    q_res = quaternion.as_float_array(quaternion.squad(qs, t_scaled, t_vector))
    
    # TODO: Make an interpolator in order to publish with custom timelimit/timestep.
    # Make a new T vector with the desired time-limit
    
    for i in range(len(t_vector)):
        new_t = t_vector[i]
        new_x = np.array(splev(new_t, tck, ext=3))
        new_q = q_res[i]

        # print(new_t, new_x, new_q)
        
        # TODO: For testing the neutral orientation of the mocap.
        # new_q = np.array([1., 0., 0., 0.])
        
        yield new_t, new_x, new_q

def generate_sampled_real_trajectory(environment_kwargs, generator_kwargs, extracts_path):
    # Getting parameters
    time_scale_offset = generator_kwargs["time_scale_offset"]
    time_scale_factor = generator_kwargs["time_scale_factor"]
    time_noise_factor = generator_kwargs["time_noise_factor"]
    control_timestep  = environment_kwargs["control_timestep"]
    time_limit = environment_kwargs["time_limit"]

    mocap_traj = [gen_traj_real(time_limit=time_limit,
                                control_timestep=control_timestep,
                                time_scale_offset=time_scale_offset,
                                time_scale_factor=time_scale_factor,
                                time_noise_factor=time_noise_factor,
                                extracts_path=extracts_path)]
    return mocap_traj




#####################################################################
### Simulated minimum-jerk trajectories ###
###########################################

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
        #        pos   vel    acc
        #yield q[:,0],q[:,1],q[:,2]
        
        yield t, q[:,0], None

###########################
# Generate randomized env #
###########################

def generate_randomized_simulated_approach(environment_kwargs, generator_kwargs):
    ## Randomizing initial position x&y on a quadcircle:
    # NOTE: Don't start too close, give the agent some time.
    r = np.random.rand() * 0.25 + 0.1
    theta = np.random.rand() * np.pi/14 + np.pi/14

    start_point = np.array([-r * np.sin(theta), -r * np.cos(theta), 0.1], dtype=np.float32)
    
    e1 = 0.02 + 0.020 * (2*np.random.rand()-1)
    e2 = 0.10 + 0.020 * (2*np.random.rand()-1)

    approach_point = np.array([e1,  0.00,  e2], dtype=np.float32)
    top_point = approach_point + np.array([0.00,  0.00,  0.2], dtype=np.float32)

    points = []
    points.append(start_point)
    points.append(approach_point)
    points.append(top_point)

    # self.params["environment_kwargs"]["time_limit"]
    # self.params["environment_kwargs"]["control_timestep"]
    time_limit = environment_kwargs["time_limit"]
    control_timestep = environment_kwargs["control_timestep"]

    times = [time_limit/2, time_limit/3]

    mocap_traj = []
    for i in range(len(points)-1):
        mocap_traj.append(gen_traj_min_jerk(points[i], points[i+1], times[i], control_timestep))
    
    return start_point, mocap_traj
