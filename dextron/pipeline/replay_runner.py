from digideep.pipeline import Runner
from digideep.utility.profiling import KeepTime
from digideep.utility.logging import logger
import gc, time

from digideep.environment import Explorer
# from from digideep.utility.toolbox import get_class
from digideep.memory.ringbuffer import Memory

from digideep.environment.dmc2gym.registration import EnvCreator
from dextron.zoo.hand_env.hand import grasp

class PlaybackRunner (Runner):
    def override(self):
        repeal = self.session.args["repeal"]
        self.session.dump_repeal(repeal)

        # reset = repeal.get("reset", "False")
        extra_epochs = repeal.get("extra_epochs", 0)

        self.params["runner"]["n_epochs"] += extra_epochs
        
        
        
    def train_cycle(self):
        # 1. Do experiment on "train"
        with KeepTime("train"):
            chunk = self.explorer["train"].update()
            with KeepTime("store"):
                self.memory["train"].store(chunk)
        # 2. Store result of "train"
        # with KeepTime("store/train"):
        #     self.memory.store(chunk)
        
        # Syncing strategy for normalizers:
        #   1) Read normalizations from "train".
        #   2) Update these in other explorers.
        #   3) Update "train" back.
        
        """DEMO MODE
        In the demo mode, we do a demonstration and then store the resulting trajectory.
        To be in demo mode, also pay attention to the following:
        1. In the sampler, use demonstrator actions.
        2. In the params file, define a memory for demos.
        3. In the runner, uncomment the following part of the loop.
        """
        # 3. Do a demo and store the trajectory
        with KeepTime("demo"):
            # if (self.memory["demo"].full and (self.state["i_cycle"] % 3 == 0)) or (not self.memory["demo"].full):
            self._sync_normalizations(source_explorer="train", target_explorer="demo")
            chunk = self.explorer["demo"].update()
            self._sync_normalizations(source_explorer="demo", target_explorer="train")
            with KeepTime("store"):
                self.memory["demo"].store(chunk)
            

        """REPLAY MODE
        Here we do a demonstration, but we do not store the trajectory of the demonstrator.
        Then we reset an explorer to the current state of the demonstrator.
        Then we do a few steps from that state, and store the whole trajectory.

        .. note:: Synchronization of normilizers:
            There different thing we can do about synchronization of normalization parameters of the 
            explorers. The most natural thing is to use all explorers to update the normalization weights
            generally. But since demo is almost doing the same thing all the time, it might be better
            to not use demo explorer too much for updating the global weights.
        """
        # # 3. Do a demo and then replay from where we left in "demo"
        # with KeepTime("demo"):
        #     self._sync_normalizations(source_explorer="train", target_explorer="demo")
        #     chunk = self.explorer["demo"].update()
        #     # self._sync_normalizations(source_explorer="demo", target_explorer="train")
        #     # self.memory["replay"].store(chunk)
        #
        #     ## We cannot train on demo data, so better off not to store them at all.
        #     ## If we need to store them for any reasons, use "sampler_list" together
        #     ## with a pre-sampler function to discard "demo" data before sampling.
        #     # self.memory.store(chunk)
        #
        # with KeepTime("replay"):
        #     # Note that the following also synchronizes the normalizations.
        #     self.explorer["replay"].load_state_dict(self.explorer["demo"].state_dict())
        #     chunk =  self.explorer["replay"].update()
        #     self._sync_normalizations(source_explorer="replay", target_explorer="train")
        #     self.memory["replay"].store(chunk)


        # 5. Update Agent
        with KeepTime("update"):
            for agent_name in self.agents:
                with KeepTime(agent_name):
                    self.agents[agent_name].update()

                    
    def custom(self):
        # 0. Load parameters from 
        repeal = self.session.args["repeal"]
        self.session.dump_repeal(repeal)

        entrypoint = repeal.get("entrypoint", "custom")
        if entrypoint == "custom":
            ##################################################
            ### FIXING DISCREPENCIES: DOES NOT WORK THOUGH ###
            # 1. Build the custom explorer
            # # Nullify both norm and vect wrappers.
            # self.params["env"]["norm_wrappers"] = []
            # self.params["env"]["vect_wrappers"] = []

            # NOTE: Workaround, add the missing keys in the hand environment per se.
            # # Add excluded observations again
            # task_kwargs = self.params["env"]["register_args"]["kwargs"]["dmcenv_creator"].task_kwargs
            # environment_kwargs = self.params["env"]["register_args"]["kwargs"]["dmcenv_creator"].environment_kwargs
            # visualize_reward = self.params["env"]["register_args"]["kwargs"]["dmcenv_creator"].visualize_reward
            # task_kwargs["exclude_obs"] = []
            # self.params["env"]["register_args"]["kwargs"]["dmcenv_creator"] = EnvCreator(
            #     grasp,
            #     task_kwargs=task_kwargs,
            #     environment_kwargs=environment_kwargs,
            #     visualize_reward=False)
            ##################################################
            

            explorer_args = {"mode":"custom",
                            "env":self.params["env"],
                            "do_reset":False,
                            "final_action":False,
                            "warm_start":0, # In less than "warm_start" steps the agent will take random actions. 
                            "num_workers":repeal.get("num_workers", 1),
                            "deterministic":True,
                            "n_steps":repeal.get("n_steps", 1), # Number of steps to take a step in the environment
                            "n_episodes":None, # Do not limit # of episodes
                            "win_size":20, # Number of episodes to episode reward for report
                            "render":repeal.get("render", "False"),
                            "render_delay":0,
                            "seed":95,
                            "extra_env_kwargs":{"mode":"custom", "allow_demos":False}
                            }

            self.explorer["custom"] = Explorer(self.session, agents=self.agents, **explorer_args)
            self._sync_normalizations(source_explorer="train", target_explorer="custom")
            # self.explorer["custom"].load_state_dict(self.explorer["train"].state_dict())
            self.explorer["custom"].reset()


            #########################
            ### Temporary Section ###
            #########################
            # print("--------------------------------------------")
            # state_dict = self.explorer["train"].state_dict()
            # keys = ["digideep.environment.wrappers.normalizers:VecNormalizeObsDict", "digideep.environment.wrappers.normalizers:VecNormalizeRew"]
            # state_dict_mod = {}
            # for k in keys:
            #     if k in state_dict["envs"]:
            #         state_dict_mod[k] = state_dict["envs"][k]
            # print(state_dict_mod)
            # exit()
            #########################


            # Reset runner so it can run from scratch.
            self.state["i_epoch"] = 0
            self.params["runner"]["n_epochs"] = repeal["number_epochs"]

            # 2. Build the custom memory
            # Remove unnecessary memories or clean their states. Create memories you may want.
            memory_list = list(self.memory.keys())
            for memory_name in memory_list:
                del self.memory[memory_name]
            # Build new memory that matches
            memory_args = {"name":"demo",
                        "keep_old_checkpoints":repeal.get("keep_old_checkpoints", False),
                        "chunk_sample_len":repeal.get("n_steps", 1),
                        "buffer_chunk_len":repeal["number_epochs"] * self.params["runner"]["n_cycles"],
                        "overrun":1}
            self.memory["custom"] = Memory(self.session, mode="custom", **memory_args)

            # 3. Run the experiment
            try:
                while (self.state["i_epoch"] < self.params["runner"]["n_epochs"]) and not self.termination_check():
                    self.state["i_cycle"] = 0
                    while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                        with KeepTime("/"):
                            with KeepTime("custom"):
                                chunk = self.explorer["custom"].update()
                                
                                with KeepTime("store"):
                                    self.memory["custom"].store(chunk)

                        self.state["i_cycle"] += 1
                        # End of Cycle
                    self.state["i_epoch"] += 1
                    self.monitor_epoch()
                    self.iterations += 1

                    self.log()
                    # Free up memory from garbage.
                    gc.collect() # Garbage Collection

            except (KeyboardInterrupt, SystemExit):
                logger.fatal('Operation stopped by the user ...')
            finally:
                self.finalize()
        
        ###################################################################
        ###################################################################
        elif entrypoint=="simonreal":
            from digideep.environment.dmc2gym.registration import EnvCreator
            from dextron.zoo.hand_env.hand import grasp


            task_kwargs = {"generator_type":"real",
                           "generator_args":{"time_scale_offset":0.5,
                                             "time_scale_factor":2.5,
                                             "time_noise_factor":0.8,
                                             "time_staying_more":20,
                                             "extracts_path":"./workspace/extracts",
                                             "database_filename":"./workspace/parameters/for-input/session_20201011162549_sweet_cray.csv"},
                           "random":None,
                           "pub_cameras":False,
                           "reward_type":"reward/20",
                           "exclude_obs":["rel_obj_hand_dist","rel_obj_hand","distance2","closure","timestep"]}
            
            # visualize_reward=True
            environment_kwargs = {"time_limit":10.0, "control_timestep":0.02}
            self.params["env"]["name"] = "CustomDMCHandGrasp-v1"
            self.params["env"]["register_args"] = {"id":"CustomDMCHandGrasp-v1",
                                                   "entry_point":"digideep.environment.dmc2gym.wrapper:DmControlWrapper",
                                                   "kwargs":{'dmcenv_creator':EnvCreator(grasp,
                                                                                         task_kwargs=task_kwargs,
                                                                                         environment_kwargs=environment_kwargs,
                                                                                         visualize_reward=True),
                                                             'flat_observation':False,
                                                             'observation_key':"agent"}
                                                  }



            from digideep.environment import MakeEnvironment
            menv = MakeEnvironment(session=None, mode=None, seed=1, **self.params["env"])
            self.params["env"]["config"] = menv.get_config()



            
            self.params["explorer"]["eval"]["env"] = self.params["env"]



            print("OURRRSELVES:", self.params["explorer"]["eval"]["env"]["register_args"]["kwargs"]["dmcenv_creator"].task_kwargs["generator_type"])
            
            self.explorer["eval"]  = Explorer(self.session, agents=self.agents, **self.params["explorer"]["eval"])
            # self.explorer["custom"].load_state_dict(self.explorer["train"].state_dict())
            self.explorer["eval"].reset()



            import glfw
            glfw.init()

            try:
                self._sync_normalizations(source_explorer="train", target_explorer="eval")
                self.explorer["eval"].reset()
                while True:
                    # Cycles
                    self.state["i_cycle"] = 0
                    while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                        with KeepTime("/"):
                            # 1. Do Experiment
                            with KeepTime("eval"):
                                self.explorer["eval"].update()
                        self.log()
                        self.state["i_cycle"] += 1
                    # Log
            except (KeyboardInterrupt, SystemExit):
                logger.fatal('Operation stopped by the user ...')
            finally:
                self.finalize(save=False)










# env.physics.get_state()
# env.physics.set_state(physics_state)

## Considering non-determinism and warmstart issue (s: 4):
#   https://github.com/deepmind/dm_control/issues/64
#   https://github.com/deepmind/dm_control/issues/65
# We won't consider it for now.





