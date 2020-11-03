import numpy as np
from copy import deepcopy

from digideep.utility.toolbox import get_class
from digideep.utility.logging import logger
# from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

from digideep.agent.agent_base import AgentBase
from scipy import stats


import torch
import torch.nn as nn

# Should output the probability of success
class PolicyModel(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(PolicyModel, self).__init__()
        init_w=3e-3
        
        # self.action_scale = 1
        
        self.linear1 = nn.Linear(state_size,  hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, states):
        # x = torch.cat([states, action], 1)
        x = states
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # x = torch.tanh(self.linear3(x)) * self.action_scale
        x = self.linear3(x)
        return x


class Controller(AgentBase):
    """
    Tha Controller is a model-based "naive" controller which has access to any
    required underlying parameter/state that RL does not have access to necessarily.
    """
    def __init__(self, session, memory, **params):
        super(Controller, self).__init__(session, memory, **params)

        act_space = self.params["methodargs"]["act_space"]
        assert act_space["typ"] == "Box", "We only support continuous actions in this demonstrator."
        # state_size  = self.params["obs_space"]["dim"][0]
        self.action_size = act_space["dim"] if np.isscalar(act_space["dim"]) else act_space["dim"][0]
        self.device = self.session.get_device()

        # LOADING the policy NN model. Load the states by state_dict.
        self.model = PolicyModel(state_size=48, hidden_size=256)
        self.model.to(self.device)
        state_dict = torch.load(f"./workspace/trained_models/{self.params['methodargs']['modelname']}.pt")
        self.model.load_state_dict(state_dict)

        # self.softmaxfn = nn.Softmax(dim=1)

        # self.sample_type = self.params['methodargs']['sample_type']
        # self.action_range = self.params['methodargs']['action_range']
        # self.action_resolution = self.params['methodargs']['action_resolution']
        



    ###############
    ## SAVE/LOAD ##
    ###############
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass
    ############################################################
    # def reset_hidden_state(self, num_workers):
    #     """
    #     "hidden_state" should be a dict of lists. It SHOULDN'T be a list of dicts (like "info").
    #     """
    #     h = {"time_step":np.zeros(shape=(num_workers, 1), dtype=np.float32),
    #          "initial_distance":np.ones(shape=(num_workers, 1), dtype=np.float32),
    #          "controller_gain":np.ones(shape=(num_workers, 1), dtype=np.float32),
    #          "controller_thre":np.ones(shape=(num_workers, 1), dtype=np.float32)}
    #     return h

    # def _lazy_initialization(self, observations):
    #     hidden_state["initial_state"] = initial_state

    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        num_workers = masks.shape[0]
        
        # 1. Prepare input to the model. Treat workers as a batch. Make sure the dimensions are right.
        position_np = observations["/agent/position"]
        velocity_np = observations["/agent/velocity"]
        states_tensor = torch.from_numpy(np.concatenate([position_np,velocity_np], axis=1)).to(self.device).float()

        with torch.no_grad():
            actions = self.model(states_tensor)
        
        actions = actions.cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)

        ## actions = np.zeros(shape=(num_workers, self.action_size))
        results = dict(actions=actions, hidden_state={}, artifacts={})
        return results

    def step(self):
        # monitor("/update/loss", Loss.item())
        self.state["i_step"] += 1

    def update(self):
        ## Update the networks for n times
        ## for i in range(self.params["methodargs"]["n_update"]):
        ##     self.step()
        pass
