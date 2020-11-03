from digideep.environment.common.vec_env import VecEnvWrapper
from digideep.environment.common.monitor import ResultsWriter
import numpy as np
import time

from digideep.utility.logging import logger
from dextron.wrappers.graph import Histogram
from dextron.wrappers.graph import Scatter

from digideep.pipeline.session import writers

from copy import deepcopy

class VecSuccessLogger(VecEnvWrapper):
    def __init__(self, venv, mode, session_state=None, threshold=0., interval=100, info_keys=["/rand"], obs_keys=["/parameters"], num_workers=None):
        VecEnvWrapper.__init__(self, venv)
        self.session_state = session_state
        
        self.eprets = None
        self.eplens = None
        self.tstart = time.time()

        # Observations are one step old, so should be the infos!
        self.infos = None
        self.obs = None

        if self.session_state:
            self.filename = self.session_state['path_session']
            # print(">>>> Success filename set to:", self.filename)
        else:
            self.filename = ""

        
        self.results_writer = None
        self.num_workers = num_workers

        self.threshold = threshold
        self.interval = interval

        self.key_root_infos = info_keys
        self.key_root_obs = obs_keys

        self.stats = {"episodes":0, "episodes_failure":0,
                    #   "scatter":{
                    #       "offset_noise_2d": Scatter("Offset Noise 2D")
                    #   },
                      "scatter": {
                          "offset_noise_2d": [[],[]],
                          "controller_thre": [],
                          "controller_gain": [],
                          "randomized_duration": [],
                          "r": [],
                          "subject_id": [],
                          "starting_id": []
                      },
                      "histogram":{
                        #   "controller_thre": Histogram("Controller Threshold", limit=[0.05,0.25], bins=4),
                        #   "controller_gain": Histogram("Controller Gain", limit=[0,10], bins=10),

                        #   "randomized_duration": Histogram("Randomized Duration", limit=[0.0,10.0], bins=10),
                        #   "r": Histogram("Rewards", limit=[0,55], bins=11),
                        #   "subject_id": Histogram("Subject ID", categories=["01","02","03","04","05","06","07","08","09"]),
                          "rotation_style": Histogram("Rotation Style", categories=["nl","nr","fl","fr"]),
                        #   "starting_id": Histogram("Starting ID", categories=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16"]),

                          "worker_id": Histogram("Worker ID", categories=list(range(num_workers)))
                      }}
    
    def update_graph(self, epinfo, success = False):
        # TODO: Create a histogram class. Automatically create a dictionary with proper-sized bins. Automatically add each new item to a bin.
        #       Give access to the data for easy plotting with "Pyasciigraph".
        
        # Update histograms
        base_filename = epinfo["/rand/filename"]

        # writers[0].add_text("Filename", base_filename)

        subject_id, rotation_style, starting_id = base_filename.split("_")
        # self.stats["histogram"]["subject_id"].add(subject_id, success)
        self.stats["histogram"]["rotation_style"].add(rotation_style, success)
        # self.stats["histogram"]["starting_id"].add(starting_id, success)
        
        # self.stats["histogram"]["r"].add(epinfo["r"], success)
        # self.stats["histogram"]["l"].add(epinfo["l"])
        
        # self.stats["histogram"]["randomized_duration"].add(epinfo["/rand/randomized_time"], success)

        self.stats["histogram"]["worker_id"].add(epinfo["/worker"], success)

        # self.stats["histogram"]["controller_thre"].add(epinfo["/parameters/controller_thre"], success)
        # self.stats["histogram"]["controller_gain"].add(epinfo["/parameters/controller_gain"], success)

        # Update scatter plots
        # if success:
        #     offset_noise_2d = epinfo["/rand/offset_noise_2d"]
        #     self.stats["scatter"]["offset_noise_2d"].add(offset_noise_2d[0], offset_noise_2d[1])
        self.stats["scatter"]["r"].append(epinfo["r"])
        if success:    
            offset_noise_2d = epinfo["/rand/offset_noise_2d"]
            self.stats["scatter"]["offset_noise_2d"][0].append(offset_noise_2d[0])
            self.stats["scatter"]["offset_noise_2d"][1].append(offset_noise_2d[1])
            
            self.stats["scatter"]["randomized_duration"].append(epinfo["/rand/randomized_time"])
            self.stats["scatter"]["controller_thre"].append(epinfo["/parameters/controller_thre"])
            self.stats["scatter"]["controller_gain"].append(epinfo["/parameters/controller_gain"])

            self.stats["scatter"]["subject_id"].append(int(subject_id))
            self.stats["scatter"]["starting_id"].append(int(starting_id))
            




    def extract_keys(self, D, extra_keys, key_root):
        for k in D:
            for root in key_root:
                if k.startswith(root):
                    extra_keys.append(k)
                    break
        return extra_keys
    def add_keys(self, target, resource, index, key_root_list):
        for k in resource:
            for root in key_root_list:
                if k.startswith(root):
                    target[k] = resource[k][index]
                    break


    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros((self.num_envs, 1), 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        if not self.results_writer:
            extra_keys=['/worker']
            extra_keys = self.extract_keys(infos, extra_keys=extra_keys, key_root=self.key_root_infos)
            extra_keys = self.extract_keys(obs, extra_keys=extra_keys, key_root=self.key_root_obs)

            self.results_writer = ResultsWriter(self.filename, header={'t_start': self.tstart}, extra_keys=extra_keys)

        # print("obs termination:", obs[0][])
        # print()
        # print(rews)
        # print("rews.shape:", rews.shape)
        # print("self.eprets.shape:", self.eprets.shape)
        # print(self.eprets)
        # print()

        self.eprets += rews
        self.eplens += 1
        
        for (i, (done, ret, eplen)) in enumerate(zip(dones, self.eprets, self.eplens)):
            if done:
                # TODO: We may put a threshold on the return, and record only trajectories that
                #       produce a "return > threshold".

                epinfo = {'r': ret[0], 'l': eplen, 't': time.time() - self.tstart, '/worker': i}
                
                # Using infos that are one step old, just to be like the observations.
                # self.add_keys(epinfo, self.infos, i, self.key_root_infos)
                self.add_keys(epinfo, infos, i, self.key_root_infos)
                self.add_keys(epinfo, self.obs, i, self.key_root_obs)

                
                self.stats["episodes"] += 1
                # print(f"------ {ret} --- {self.threshold}")

                success = ret[0] >= self.threshold
                if not success:
                    # print("failure case")
                    self.stats["episodes_failure"] += 1
                else:
                    self.results_writer.write_row(epinfo)
                    # print("+++++++++++++++ success case recorded +++++++++++++++")
                    # print(epinfo)
                    # print()
                
                self.update_graph(epinfo, success)
                

                self.log()
                self.plot_stats()

                # Acts by reference!
                self.eprets[i] = 0
                self.eplens[i] = 0
        
        # self.infos = deepcopy(infos)
        self.obs = obs
        return obs, rews, dones, infos

    def log(self):
        if (self.stats["episodes"] % self.interval == 0):
            success = self.stats["episodes"] - self.stats["episodes_failure"]
            overall = self.stats["episodes"]
            success_rate = (1 - float(self.stats["episodes_failure"]) / float(self.stats["episodes"])) * 100
            logger.warn("Success rate is: {}/{} = {:4.2f}".format(success, overall, success_rate))
    
    def plot_stats(self):
        if (self.stats["episodes"] % self.interval == 0):
            # Plot histograms
            for h in self.stats["histogram"]:
                self.stats["histogram"][h].plot()



            writers[0].add_histogram("Offset Noise X", np.array(self.stats["scatter"]["offset_noise_2d"][0]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Offset Noise Y", np.array(self.stats["scatter"]["offset_noise_2d"][1]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Reward", np.array(self.stats["scatter"]["r"]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Duration", np.array(self.stats["scatter"]["randomized_duration"]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Controller Threshold", np.array(self.stats["scatter"]["controller_thre"]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Controller Gain", np.array(self.stats["scatter"]["controller_gain"]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Subject ID", np.array(self.stats["scatter"]["subject_id"]), global_step=self.stats["episodes"])
            writers[0].add_histogram("Starting ID", np.array(self.stats["scatter"]["starting_id"]), global_step=self.stats["episodes"])

            # # Plot scatters
            # for s in self.stats["scatter"]:
            #     self.stats["scatter"][s].plot()


