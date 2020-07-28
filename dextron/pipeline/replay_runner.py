from digideep.pipeline import Runner
from digideep.utility.profiling import KeepTime
from digideep.utility.logging import logger
import gc, time


class PlaybackRunner (Runner):
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

                    


# env.physics.get_state()
# env.physics.set_state(physics_state)

## Considering non-determinism and warmstart issue (s: 4):
#   https://github.com/deepmind/dm_control/issues/64
#   https://github.com/deepmind/dm_control/issues/65
# We won't consider it for now.
