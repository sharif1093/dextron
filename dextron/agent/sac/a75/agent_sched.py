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

class AgentSchedule(AgentBase):
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
        super(AgentSchedule, self).__init__(session, memory, **params)

        self.device = self.session.get_device()

        # Set the Policy
        # policyclass = get_class(self.params["policyname"])
        self.policy = Policy(device=self.device, **self.params["policyargs"])
        
        # Set the optimizer (+ schedulers if any)
        optimclass_value = get_class(self.params["optimname_value"])
        optimclass_softq = get_class(self.params["optimname_softq"])
        optimclass_actor = get_class(self.params["optimname_actor"])
        
        self.optimizer = {}
        # self.optimizer["image"] = optimclass_value(self.policy.model["value"].parameters(), **self.params["optimargs_value"])

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
        
        initial = self.params["sampler_args"]["scheduler_start"]
        interval = self.params["sampler_args"]["scheduler_steps"]
        decay = self.params["sampler_args"]["scheduler_decay"]

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

    def sample(self):
        with KeepTime("sampler"):
            info = deepcopy(self.params["sampler_args"])

            batch_size = info["batch_size"]
            b = self.scheduler.value

            demo_batch_size = int(b * batch_size)
            train_batch_size  = batch_size - demo_batch_size

            info["batch_size_dict"]= {"train":train_batch_size, "demo":demo_batch_size}

            batch = self.sampler(data=self.memory, info=info)
            return batch
    
    def fetch(self, batch):
        with KeepTime("fetch/to_torch"):
            state      = torch.from_numpy(batch["/observations"+ self.params["observation_path"]]).to(self.device)
            action     = torch.from_numpy(batch["/agents/"+self.params["name"]+"/actions"]).to(self.device)
            reward     = torch.from_numpy(batch["/rewards"]).to(self.device)
            next_state = torch.from_numpy(batch["/observations"+self.params["observation_path"]+"_2"]).to(self.device)
            masks      = torch.from_numpy(batch["/masks"]).to(self.device)
        return state, action, reward, next_state, masks

    def calculate_loss(self, state, action, reward, next_state, masks):
        raise NotImplementedError


    def step(self):
        """This function needs the following key values in the batch of memory:

        * ``/observations``
        * ``/rewards``
        * ``/agents/<agent_name>/actions``
        * ``/observations_2``

        The first three keys are generated by the :class:`~digideep.environment.explorer.Explorer`
        and the last key is added by the sampler.
        """
        
        with KeepTime("sample"):
            batch = self.sample()
        if batch is None:
            return
        



        ## Sequence of images as stacking
        #
        # shape = batch["/observations/camera"][0].shape
        # self.session.writer.add_images(tag=self.params["name"]+"_images", 
        #                                img_tensor=batch["/observations/camera"][0].reshape(shape[0],1,shape[1],shape[2]),
        #                                global_step=self.state['i_step'],
        #                                dataformats='NCHW')
        #

        # self.session.writer.add_images(tag=self.params["name"]+"_images", 
        #                                img_tensor=batch["/observations/camera"][:,1:,:,:],
        #                                global_step=self.state['i_step'],
        #                                dataformats='NCHW')
        
        # print("1. RAW AVERAGE OF OUR FIRST INSTANCE:", np.mean(batch["/observations/camera"][0]), "|     STD:", np.std(batch["/observations/camera"][0]))
        
        # print("2. RAW AVERAGE OF OUR FIRST INSTANCE:", np.mean(batch["/observations/camera"]/255.), "|     STD:", np.std(batch["/observations/camera"]/255.))

        # print(batch["/observations/camera"][0])
        # print("\n\n\n")
        
        # batch["/observations/camera"][:,1:,:,:] vs batch["/observations/camera"][:,:3,:,:]
        # Thesecond shows complete black at the very first frame since frame-stacking stackes with zero frames.
        # The first one should always show something.
        #
        ## Sequence of images as channels
        # a1 = batch["/observations/camera"][0].reshape(1, *shape)
        # a2 = batch["/observations/camera_2"][0].reshape(1, *shape)
        # c = np.concatenate([a1,a2])
        # self.session.writer.add_images(tag=self.params["name"]+"_images_0", 
        #                               img_tensor=c[:,:3,:,:],
        #                               global_step=self.state['i_step'],
        #                               dataformats='NCHW')
        #
        # self.session.writer.add_image(tag=self.params["name"]+"_images_1", 
        #                               img_tensor=batch["/observations/camera_2"][1].reshape(1, *shape),
        #                               global_step=self.state['i_step'],
        #                               dataformats='CHW')

        # batch["/observations/camera"]   = (batch["/observations/camera"]   - 16.4) / 17.0
        # batch["/observations/camera_2"] = (batch["/observations/camera_2"] - 16.4) / 17.0

        with KeepTime("fetch"):
            state, action, reward, next_state, masks = self.fetch(batch)
        
        self.calculate_loss(state, action, reward, next_state, masks)
        self.state["i_step"] += 1

        ## Sending visualizations to Tensorboard
        # self.session.writer.add_scalar('loss/actor', actor_loss.item(), self.state["i_step"])
        # self.session.writer.add_scalar('loss/softq', softq_loss.item(), self.state["i_step"])
        # self.session.writer.add_scalar('loss/value', value_loss.item(), self.state["i_step"])


    def update(self):
        # Update the networks for n times
        for i in range(self.params["methodargs"]["n_update"]):
            with KeepTime("scheduler"):
                self.scheduler.update()
            
            # Step
            with KeepTime("step"):
                self.step()
            
            # Update value target
            with KeepTime("targets"):
                self.policy.averager["value"].update_target()
        
        
        # ## For debugging
        # # for p, ptar in zip(self.policy.model["actor"].parameters(), self.policy.model["actor_target"].parameters()):
        # #     print(p.mean(), ptar.mean())
    
        # # for p, ptar in zip(self.policy.model["actor"].parameters(), self.policy.model["critic"].parameters()):
        # #     print(p.mean(), ptar.mean())


