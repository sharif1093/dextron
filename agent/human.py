import numpy as np
import random
from copy import deepcopy

from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

from digideep.agent import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, session, memory, **params):
        super(Trainer, self).__init__(session, memory, **params)

    ###############
    ## SAVE/LOAD ##
    ###############
    def state_dict(self):
        return None
    def load_state_dict(self, state_dict):
        pass
    ############################################################
    def reset_hidden_state(self, num_workers):
        # np.full((num_workers,1), fill_value=np.nan, dtype=np.float32)
        h = {"time":np.random.rand(num_workers, 1)}
        return h

    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        """
        Arguments:
            observations: Gets the observation from the environment. This is important
                          to decide on the next action.
            hidden_state: Carries the important recurrent information from previous
                          running of the action_generator for the next run.
            masks: Indicates the end of episodes.
        """
        # with KeepTime("/explore/step/prestep/gen_action/to_torch"):
        # observations = torch.from_numpy(observations).to(self.device)
        # hidden_state = torch.from_numpy(hidden_state).to(self.device)
        # masks = torch.from_numpy(masks).to(self.device)

        ## TODO: Put all of them into one dictionary
        ####### action, values, hidden_state, action_log_p
        
        num_workers = observations.shape[0]

        # print("Time =", observations[0][-1])

        a = np.random.rand(num_workers, 3) * 0.01
        h = {"time":np.random.rand(num_workers, 1)}
        return {'actions':a, 'hidden_state':h, 'artifacts':{}}


    def step(self):
        # monitor("/update/loss", Loss.item())
        self.state["i_step"] += 1

    def update(self):
        ## Update the networks for n times
        ## for i in range(self.params["methodargs"]["n_update"]):
        ##     self.step()
        pass
        

        
