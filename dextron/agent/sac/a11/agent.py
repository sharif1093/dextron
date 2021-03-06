"""
This implementation is mainly adopted from `RL-Adventure-2 <https://github.com/higgsfield/RL-Adventure-2>`__.
"""

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from digideep.utility.toolbox import get_class
from digideep.utility.logging import logger
from digideep.utility.profiling import KeepTime
from digideep.utility.monitoring import monitor

# from digideep.agent.samplers.ddpg import sampler_re
from digideep.agent.sampler_common import Compose
from digideep.agent.agent_base import AgentBase
from dextron.utils import Scheduler
from .policy import Policy

# torch.utils.backcompat.broadcast_warning.enabled = True

class Agent(AgentBase):
    """This is an implementation of the Soft Actor Critic (`SAC <https://arxiv.org/abs/1801.01290>`_) method.
    Here the modified version of `SAC https://arxiv.org/abs/1812.05905`_ is not considered.
    
    Args:
        name: The agent's name.
        type: The type of this class which is ``digideep.agent.SAC``.
        methodargs (dict): The parameters of the SAC method.
        sampler:
        
        
    #     policyname: The name of the policy which can be ``digideep.agent.policy.soft_stochastic.Policy`` for normal SAC.
    #     policyargs: The arguments for the policy.
    #     noisename: The noise model name.
    #     noiseargs: The noise model arguments.
    #     optimname: The name of the optimizer.
    #     optimargs: The arguments of the optimizer.
        
    # The elements in the ``methodargs`` are:

    # * ``n_update``: Number of times to perform SAC step.
    # * ``gamma``: Discount factor :math:`\gamma`.
    # * ``clamp_return``: The clamp factor. One option is :math:`1/(1-\gamma)`.
    

    """

    def __init__(self, session, memory, **params):
        super(Agent, self).__init__(session, memory, **params)

        self.device = self.session.get_device()

        # Set the Policy
        # policyclass = get_class(self.params["policyname"])
        self.policy = Policy(device=self.device, **self.params["policyargs"])
        
        # Set the optimizer (+ schedulers if any)
        optimclass_value = get_class(self.params["optimname_value"])
        optimclass_softq = get_class(self.params["optimname_softq"])
        optimclass_actor = get_class(self.params["optimname_actor"])
        
        self.optimizer = {}
        self.optimizer["value"] = optimclass_value(self.policy.model["value"].parameters(), **self.params["optimargs_value"])
        self.optimizer["softq"] = optimclass_softq(self.policy.model["softq"].parameters(), **self.params["optimargs_softq"])
        self.optimizer["actor"] = optimclass_actor(self.policy.model["actor"].parameters(), **self.params["optimargs_actor"])

        self.criterion = {}
        self.criterion["value"] = nn.MSELoss()
        self.criterion["softq"] = nn.MSELoss()
        
        # Build the sampler from sampler list:
        sampler_list = [get_class(k) for k in self.params["sampler_list"]]
        self.sampler = Compose(sampler_list)

        # noiseclass = get_class(self.params["noisename"])
        # self.noise = noiseclass(**self.params["noiseargs"])

        self.state["i_step"] = 0
        
        interval = self.params["sampler_args"]["scheduler_steps"]
        decay = self.params["sampler_args"]["scheduler_decay"]
        initial = self.params["sampler_args"]["scheduler_start"]

        self.scheduler = Scheduler(initial, interval, decay)

    ###############
    ## SAVE/LOAD ##
    ###############
    # TODO: Also states of optimizers, noise, etc.
    def state_dict(self):
        return {'state':self.state, 'policy':self.policy.model.state_dict()}
    def load_state_dict(self, state_dict):
        self.policy.model.load_state_dict(state_dict['policy'])
        self.state.update(state_dict['state'])
    ############################################################
    
    def action_generator(self, observations, hidden_state, masks, deterministic=False):
        """This function computes the action based on observation, and adds noise to it if demanded.

        Args:
            deterministic (bool): If ``True``, the output would be merely the output from the actor network.
            Otherwise, noise will be added to the output actions.
        
        Returns:
            dict: ``{"actions":...,"hidden_state":...}``

        """
        observation_path = self.params.get("observation_path", "/agent")
        observations_ = observations[observation_path].astype(np.float32)
        
        observations_ = torch.from_numpy(observations_).to(self.device)
        action = self.policy.generate_actions(observations_, deterministic=deterministic)
        action = action.cpu().numpy().astype(np.float32)

        # if not deterministic:
        #     action = self.noise(action)

        results = dict(actions=action, hidden_state=hidden_state)
        return results


    def step(self):
        """This function needs the following key values in the batch of memory:

        * ``/observations``
        * ``/rewards``
        * ``/agents/<agent_name>/actions``
        * ``/observations_2``

        The first three keys are generated by the :class:`~digideep.environment.explorer.Explorer`
        and the last key is added by the sampler.
        """
        with KeepTime("sampler"):
            info = deepcopy(self.params["sampler_args"])

            batch_size = info["batch_size"]
            b = self.scheduler.value

            demo_batch_size = int(b * batch_size)
            train_batch_size  = batch_size - demo_batch_size

            info["batch_size_dict"]= {"train":train_batch_size, "demo":demo_batch_size}

            batch = self.sampler(data=self.memory, info=info)
            if batch is None:
                return


        with KeepTime("to_torch"):
            # ['/obs_with_key', '/masks', '/agents/agent/actions', '/agents/agent/hidden_state', '/rewards', '/obs_with_key_2', ...]
            

            # for k in batch:
            #     print(k, "dtype:", batch[k].dtype)
            # exit()
            
            # Keys:
            #   /observations/agent dtype: float32
            #   /observations/demonstrator/distance dtype: float32
            #   /observations/demonstrator/hand_closure dtype: float32
            #   /observations/status/is_training dtype: uint8
            #   /masks dtype: float32
            #   /agents/agent/actions dtype: float32
            #   /agents/agent/hidden_state dtype: float32
            #   /agents/demonstrator/actions dtype: float32
            #   /agents/demonstrator/hidden_state/time_step dtype: float32
            #   /agents/demonstrator/hidden_state/initial_distance dtype: float32
            #   /rewards dtype: float32
            #   /obs_with_key dtype: float32
            #   /obs_with_key_2 dtype: float32
            
            #   /observations/camera_2 dtype: float32
            #   /observations/camera dtype: uint8

            # state      = torch.from_numpy(batch["/obs_with_key"]).to(self.device)
            # next_state = torch.from_numpy(batch["/obs_with_key_2"]).to(self.device)

            state      = torch.from_numpy(batch["/observations/camera"]).to(self.device)
            action     = torch.from_numpy(batch["/agents/"+self.params["name"]+"/actions"]).to(self.device)
            reward     = torch.from_numpy(batch["/rewards"]).to(self.device)
            next_state = torch.from_numpy(batch["/observations/camera_2"]).to(self.device)
            # masks      = torch.from_numpy(batch["/masks"]).to(self.device).view(-1)
            masks      = torch.from_numpy(batch["/masks"]).to(self.device)

            # print("state =", state.shape)
            # print("action =", action.shape)
            # print("reward =", reward.shape)
            # print("next_state =", next_state.shape)
            # print("masks =", masks.shape)
            # exit()



            ## print(">>>>>> state shape =", state.shape)
            ## print(">>>>>> next_state shape =", next_state.shape)

        with KeepTime("loss"):
            expected_q_value = self.policy.model["softq"](state, action)
            expected_value = self.policy.model["value"](state)
            new_action, log_prob, z, mean, log_std = self.policy.evaluate_actions(state)

            target_value = self.policy.model["value_target"](next_state)
            
            next_q_value = reward + masks * float(self.params["methodargs"]["gamma"]) * target_value
            softq_loss = self.criterion["softq"](expected_q_value, next_q_value.detach())

            expected_new_q_value = self.policy.model["softq"](state, new_action)
            next_value = expected_new_q_value - log_prob
            value_loss = self.criterion["value"](expected_value, next_value.detach())

            log_prob_target = expected_new_q_value - expected_value
            # TODO: Apperantly the calculation of actor_loss is problematic: none of its ingredients have gradients! So backprop does nothing.
            actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
            
            mean_loss = float(self.params["methodargs"]["mean_lambda"]) * mean.pow(2).mean()
            std_loss  = float(self.params["methodargs"]["std_lambda"])  * log_std.pow(2).mean()
            z_loss    = float(self.params["methodargs"]["z_lambda"])    * z.pow(2).sum(1).mean()

            actor_loss += mean_loss + std_loss + z_loss

        with KeepTime("optimization"):
            self.optimizer["softq"].zero_grad()
            softq_loss.backward()
            self.optimizer["softq"].step()

            self.optimizer["value"].zero_grad()
            value_loss.backward()
            self.optimizer["value"].step()

            self.optimizer["actor"].zero_grad()
            actor_loss.backward()
            self.optimizer["actor"].step()
            
            

        monitor("/update/loss/actor", actor_loss.item())
        monitor("/update/loss/softq", softq_loss.item())
        monitor("/update/loss/value", value_loss.item())

        # for key,item in locals().items():
        #     if isinstance(item, torch.Tensor):
        #         # print("item =", type(item))
        #         print(key, ":", item.shape)
        # print("-----------------------------")

        self.state["i_step"] += 1

    def update(self):
        self.scheduler.update()
        # Update the networks for n times
        for i in range(self.params["methodargs"]["n_update"]):
            with KeepTime("step"):
                self.step()
        
        with KeepTime("targets"):
            # Update value target
            self.policy.averager["value"].update_target()
        
        # ## For debugging
        # # for p, ptar in zip(self.policy.model["actor"].parameters(), self.policy.model["actor_target"].parameters()):
        # #     print(p.mean(), ptar.mean())
    
        # # for p, ptar in zip(self.policy.model["actor"].parameters(), self.policy.model["critic"].parameters()):
        # #     print(p.mean(), ptar.mean())


