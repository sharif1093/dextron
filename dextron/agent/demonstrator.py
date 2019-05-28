import numpy as np
import random
from copy import deepcopy

from digideep.utility.toolbox import get_class
from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

from digideep.agent.base import AgentBase


_DISTANCE_CRITICAL = 0.15
_OPEN_HAND_CLOSURE = 0.2
_CLOSE_HAND_CLOSURE = 0.8
_GRASPER_GAIN = 2


class NaiveController(AgentBase):
    """
    Tha NaiveController is a model-based "naive" controller which has access to any
    required underlying parameter/state that RL does not have access to necessarily.
    """
    def __init__(self, session, memory, **params):
        super(NaiveController, self).__init__(session, memory, **params)

        act_space = self.params["methodargs"]["act_space"]
        assert act_space["typ"] == "Box", "We only support continuous actions in this demonstrator."
        # state_size  = self.params["obs_space"]["dim"][0]
        self.action_size = act_space["dim"] if np.isscalar(act_space["dim"]) else act_space["dim"][0]

    ###############
    ## SAVE/LOAD ##
    ###############
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass
    ############################################################
    def reset_hidden_state(self, num_workers):
        """
        "hidden_state" should be a dict of lists. It SHOULDN'T be a list of dicts (like "info").
        """
        h = {"time_step":np.zeros(shape=(num_workers, 1)),
             "initial_distance":np.ones(shape=(num_workers, 1))}
        return h

    # def _lazy_initialization(self, observations):
    #     hidden_state["initial_state"] = initial_state

    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        """
        Arguments:
            observations: Gets the observation from the environment. This is important
                          to decide on the next action.
            hidden_state: Carries the important recurrent information from previous
                          running of the action_generator for the next run.
            masks: Indicates the end of episodes.
        """
        num_workers = masks.shape[0]
        # actions = np.zeros(shape=(num_workers, self.action_size))
        
        actions = []
        for index in range(num_workers):
            is_training = bool(observations["/status/is_training"][index].item())
            if is_training:
                actions += [np.zeros(shape=(self.action_size,))]
            else:
                mask = masks[index]

                if mask == 0: # Environment was reset
                    distance = observations["/demonstrator/distance"][index]
                    print("Found a reset signal. Saving initial states. Distance = ", distance)
                    hidden_state["initial_distance"][index] = distance
                    hidden_state["time_step"][index] = 0
                    actions += [np.zeros(shape=(self.action_size,))]
                elif mask == 1:
                    ## The main body of the controller goes here:
                    # Values
                    initial_distance = hidden_state["initial_distance"][index]
                    time_step = hidden_state["time_step"][index]
                    distance = observations["/demonstrator/distance"][index]
                    hand_closure = observations["/demonstrator/hand_closure"][index]
                    # Pre-calculations
                    distance_normalized = distance / initial_distance

                    # Control Law
                    if distance_normalized >= _DISTANCE_CRITICAL:
                        # print("Now openning ...")
                        cmd = _GRASPER_GAIN * (_OPEN_HAND_CLOSURE - hand_closure)
                    elif distance_normalized < _DISTANCE_CRITICAL:
                        # print("Now closing ...")
                        cmd = _GRASPER_GAIN * (_CLOSE_HAND_CLOSURE - hand_closure)

                    # Storing action
                    actions += [[cmd]]
                    # Increasing timestep
                    hidden_state["time_step"][index] += 1

        actions = np.asarray(actions, dtype=np.float32)
        
        # Check if the action values are within the defined structure, then return.
        # self.params["methodargs"]["act_space"].validate(actions)

        results = dict(actions=actions, hidden_state=hidden_state, artifacts={})
        return results

    def step(self):
        # monitor("/update/loss", Loss.item())
        self.state["i_step"] += 1

    def update(self):
        ## Update the networks for n times
        ## for i in range(self.params["methodargs"]["n_update"]):
        ##     self.step()
        pass