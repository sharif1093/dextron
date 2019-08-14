from digideep.pipeline import Runner
from digideep.utility.profiling import KeepTime
from digideep.utility.logging import logger
import gc


class PlaybackRunner (Runner):
    def train(self):
        try:
            while self.state["i_epoch"] < self.params["runner"]["n_epochs"]:
                self.state["i_cycle"] = 0
                while self.state["i_cycle"] < self.params["runner"]["n_cycles"]:
                    with KeepTime("/"):
                        # 1. Do experiment on "train"
                        with KeepTime("train"):
                            chunk = self.explorer["train"].update()
                            self.memory["train"].store(chunk)
                        
                        # 2. Store result of "train"
                        # with KeepTime("store/train"):
                        #     self.memory.store(chunk)
                        
                        # Syncing strategy for normalizers:
                        #   1) Read normalizations from "train".
                        #   2) Update these in other explorers.
                        #   3) Update "train" back.
                        





                        # # 3. Do a demo and then replay from where we left in "demo"
                        # with KeepTime("demo"):
                        #     # TODO: If helpful, we can still take advantage of the "reward normalizer" and "observation normalizer" states of the train mode.
                        #     #       It may be useful for the "replay" mode but not for the "demo" mode were we don't need the data to be included in the
                        #     #       memory and training.
                        #     self._sync_normalizations(main_explorer="train", target_explorer="demo")
                        #     chunk = self.explorer["demo"].update()
                        #     self._sync_normalizations(main_explorer="demo", target_explorer="train")
                        #     self.memory["replay"].store(chunk)
                            

                        #     ## We cannot train on demo data, so better off not to store them at all.
                        #     ## If we need to store them for any reasons, use "sampler_list" together
                        #     ## with a pre-sampler function to discard "demo" data before sampling.
                        #     # self.memory.store(chunk)

                        # # with KeepTime("replay"):
                        # #     self.explorer["replay"].load_state_dict(self.explorer["demo"].state_dict())
                        # #     chunk =  self.explorer["replay"].update()
                        # #     self._sync_normalizations(main_explorer="replay", target_explorer="train")
                        # #     self.memory["replay"].store(chunk)
                        # #     ########### self.memory["replay"].store(chunk)











                        # 3. Do a demo and then replay from where we left in "demo"
                        with KeepTime("demo"):
                            # TODO: If helpful, we can still take advantage of the "reward normalizer" and "observation normalizer" states of the train mode.
                            #       It may be useful for the "replay" mode but not for the "demo" mode were we don't need the data to be included in the
                            #       memory and training.
                            self._sync_normalizations(main_explorer="train", target_explorer="demo")
                            chunk = self.explorer["demo"].update()
                            # self._sync_normalizations(main_explorer="demo", target_explorer="train")
                            # self.memory["replay"].store(chunk)
                        
                        
                            ## We cannot train on demo data, so better off not to store them at all.
                            ## If we need to store them for any reasons, use "sampler_list" together
                            ## with a pre-sampler function to discard "demo" data before sampling.
                            # self.memory.store(chunk)
                        
                        with KeepTime("replay"):
                            self.explorer["replay"].load_state_dict(self.explorer["demo"].state_dict())
                            chunk =  self.explorer["replay"].update()
                            self._sync_normalizations(main_explorer="replay", target_explorer="train")
                            self.memory["replay"].store(chunk)
                        
                            ########### self.memory["replay"].store(chunk)
                        
                        

                        # 5. Update Agent
                        with KeepTime("update"):
                            for agent_name in self.agents:
                                with KeepTime(agent_name):
                                    self.agents[agent_name].update()
                    self.state["i_cycle"] += 1
                # End of Cycle

                self._inc_epoch()
                # NOTE: We may save/test after each cycle or at intervals.

                # 1. Perform the test
                self.test()
                # 2. Save
                self.save()
                # 3. Log
                self.log()
                gc.collect() # Garbage Collection

        except (KeyboardInterrupt, SystemExit):
            logger.fatal('Operation stopped by the user ...')
        finally:
            logger.fatal('End of operation ...')
            self.finalize()
    
    def _sync_normalizations(self, main_explorer, target_explorer):
        state_dict = self.explorer[main_explorer].state_dict()
        
        keys = ["digideep.environment.wrappers.normalizers:VecNormalizeObsDict", "digideep.environment.wrappers.normalizers:VecNormalizeRew"]
        
        state_dict_mod = {}
        for k in keys:
            if k in state_dict["envs"]:
                state_dict_mod[k] = state_dict["envs"][k]

        self.explorer[target_explorer].envs.load_state_dict(state_dict_mod)


# env.physics.get_state()
# env.physics.set_state(physics_state)

## Considering non-determinism and warmstart issue (s: 4):
#   https://github.com/deepmind/dm_control/issues/64
#   https://github.com/deepmind/dm_control/issues/65
# We won't consider it for now.
