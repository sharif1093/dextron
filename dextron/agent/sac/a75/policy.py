"""
This implementation is mainly adopted from `RL-Adventure-2 <https://github.com/higgsfield/RL-Adventure-2>`__.

"""

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np

from digideep.utility.toolbox import get_class
from digideep.utility.logging import logger

from digideep.agent.policy_base import PolicyBase
from digideep.agent.policy_common import Averager

from dextron.agent.sac.CoordConv import CoordConv

from copy import deepcopy

class Policy(PolicyBase):
    """Implementation of a stochastic actor-critic policy for the SAC method.

    Args:
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        actor_args (dict): A dictionary of arguments for the :class:`ActorNetwork`.
        softq_args (dict): A dictionary of arguments for the :class:`SoftQNetwork`.
        value_args (dict): A dictionary of arguments for the :class:`ValueNetwork`.
        average_args (dict): A dictionary of arguments for the :class:`Averager`.
    
    Todo:
        Override the base class ``state_dict`` and ``load_state_dict`` to also save the state of ``averager``.

    """
    def __init__(self, device, **params):
        super(Policy, self).__init__(device)

        self.params = params

        # assert len(self.params["obs_space"]["dim"]) == 1, "We only support 1d observations for the SAC policy for now."

        assert self.params["act_space"]["typ"] == "Box", "We only support continuous actions in SAC policy for now."

        state_size  = self.params["obs_space"]["dim"][0]
        action_size = self.params["act_space"]["dim"] if np.isscalar(self.params["act_space"]["dim"]) else self.params["act_space"]["dim"][0]
        image_repr_size = self.params["image_repr_size"]

        
        print("image_repr_type =", self.params["image_repr_type"])
        if self.params["image_repr_type"] == "cnn":
            logger.warn("Using CNNBlock in the policy.")
            self.model["image"] = CNNBlock(num_inputs=state_size, output_size=image_repr_size)
        elif self.params["image_repr_type"] == "coordconv":
            logger.warn("Using CoordConv in the policy.")
            self.model["image"] = CoordConvBlock(num_inputs=state_size, output_size=image_repr_size)
        else:
            logger.error("The provided 'image_repr_type' is not recognized.")
            exit()

        
        if "hidden_size" in self.params:
            hidden_size = self.params["hidden_size"]
            self.model["value"] = ValueNetwork(image_repr_size, hidden_size, **self.params["value_args"])
            self.model["value_target"] = deepcopy(self.model["value"])
            self.model["softq"] = SoftQNetwork(image_repr_size, action_size, hidden_size, **self.params["softq_args"])
            self.model["actor"] = ActorNetwork(image_repr_size, action_size, hidden_size, **self.params["actor_args"])
        else:
            self.model["value"] = ValueNetwork(image_repr_size, **self.params["value_args"])
            self.model["value_target"] = deepcopy(self.model["value"])
            self.model["softq"] = SoftQNetwork(image_repr_size, action_size, **self.params["softq_args"])
            self.model["actor"] = ActorNetwork(image_repr_size, action_size, **self.params["actor_args"])


        self.averager = {}
        self.averager["value"] = Averager(self.model["value"], self.model["value_target"], **self.params["average_args"])
        
        self.model_to_gpu()
        logger("Number of parameters: <", self.count_parameters(), '>')
    
    def generate_actions(self, inputs, deterministic=False):
        """
        This function generates actions from the "actor" model.
        """
        # inputs = torch.FloatTensor(inputs).unsqueeze(0).to(device)
        with torch.no_grad():
            self.model.eval()

            state_repr = self.model["image"](inputs)

            mean, log_std = self.model["actor"](state_repr)
            std = log_std.exp()
            
            dist = distributions.Normal(mean, std)

            # If not deterministic sample the distribution, otherwise use mean or median.
            if not deterministic:
                z = dist.sample()
            else:
                z = dist.mean
            action = torch.tanh(z)
            
            self.model.train()
            return action

    def evaluate_actions(self, state, epsilon=1e-6):
        mean, log_std = self.model["actor"](state)
        std = log_std.exp()
    
        dist = distributions.Normal(mean, std)
        # NOTE: Why doesn't sample throw an error when back-propagating gradients through sample?
        #       See: https://github.com/pytorch/pytorch/issues/4620
        #       Also, rsample() does not work.
        #       The reason is that sample() detaches gradients silently without notice.
        #       It is interesting that rsample() won't work here, although detach is used on all
        #       outputs of this evaluate_actions() function.
        
        # NOTE: sample() will silently avoid back-propagation.
        #       rsample() will do reparametrization trick, and hence allows back-propagation.
        # Here, either of sample() or rsample().detach() should be used. rsample() won't work.
        # The gradients are propagated though `dist.log_prob(z)` though.

        z = dist.sample()
        # z = dist.rsample().detach()
        
        action = torch.tanh(z)
    
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
    
        return action, log_prob, z, mean, log_std




def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
def init_easy(gain=1, bias=0):
    def _f(module):
        return init(module=module, weight_init=nn.init.orthogonal_, bias_init=lambda x: nn.init.constant_(x, bias), gain=gain)
    return _f

def init_cc(module, weight_init, bias_init, gain=1):
    weight_init(module.conv.weight.data, gain=gain)
    bias_init(module.conv.bias.data)
    return module
def init_easy_cc(gain=1, bias=0):
    def _f(module):
        return init_cc(module=module, weight_init=nn.init.orthogonal_, bias_init=lambda x: nn.init.constant_(x, bias), gain=gain)
    return _f


class CNNBlock(nn.Module):
    def __init__(self, num_inputs, output_size):
        super(CNNBlock, self).__init__()
        init_ = init_easy(gain=nn.init.calculate_gain('relu'), bias=0)
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = init_(nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4))
        self.relu1 = nn.ReLU()
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=5, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = init_(nn.Conv2d(64, 128, kernel_size=4, stride=2))
        self.relu3 = nn.ReLU()
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1)
        self.conv4 = init_(nn.Conv2d(128, 96, kernel_size=3, stride=1))
        # self.conv4 = init_(nn.Conv2d(128, 32, 3, stride=1))
        self.relu4 = nn.ReLU()

        # Flatten Here
        self.linear = init_(nn.Linear(96 * 7 * 11, output_size))
        # self.linear = init_(nn.Linear(32 * 7 * 11, output_size))
        self.relu5 = nn.ReLU()
    def forward(self, inputs):
        x = inputs.float()/255.
        ## Inspect x.shape && x.dtype
        # print("input size =", x.shape)
        x = self.relu1(self.conv1(x))
        # print("after conv1 =", x.shape)
        x = self.relu2(self.conv2(x))
        # print("after conv2 =", x.shape)
        x = self.relu3(self.conv3(x))
        # print("after conv3 =", x.shape)
        x = self.relu4(self.conv4(x))
        # print("after conv4 =", x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        # print("after reshape =", x.shape)
        x = self.relu5(self.linear(x))
        # print("after linear =", x.shape)
        return x

class CoordConvBlock(nn.Module):
    def __init__(self, num_inputs, output_size):
        super(CoordConvBlock, self).__init__()
        init_ = init_easy_cc(gain=nn.init.calculate_gain('relu'), bias=0)
        init_lin_ = init_easy(gain=nn.init.calculate_gain('relu'), bias=0)
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.conv1 = init_(CoordConv(num_inputs, 32, kernel_size=8, stride=4))
        self.relu1 = nn.ReLU()
        self.conv2 = init_(CoordConv(32, 64, kernel_size=5, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = init_(CoordConv(64, 128, kernel_size=4, stride=2))
        self.relu3 = nn.ReLU()
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1)
        self.conv4 = init_(CoordConv(128, 96, kernel_size=3, stride=1))
        # self.conv4 = init_(CoordConv(128, 32, 3, stride=1))
        self.relu4 = nn.ReLU()

        # Flatten Here
        self.linear = init_lin_(nn.Linear(96 * 7 * 11, output_size))
        # self.linear = init_lin_(nn.Linear(32 * 7 * 11, output_size))
        self.relu5 = nn.ReLU()
    def forward(self, inputs):
        x = inputs.float()/255.
        ## Inspect x.shape && x.dtype
        # print("input size =", x.shape)
        x = self.relu1(self.conv1(x))
        # print("after conv1 =", x.shape)
        x = self.relu2(self.conv2(x))
        # print("after conv2 =", x.shape)
        x = self.relu3(self.conv3(x))
        # print("after conv3 =", x.shape)
        x = self.relu4(self.conv4(x))
        # print("after conv4 =", x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        # print("after reshape =", x.shape)
        x = self.relu5(self.linear(x))
        # print("after linear =", x.shape)
        return x

# class CNNBlock(nn.Module):
#     def __init__(self, num_inputs, output_size):
#         super(CNNBlock, self).__init__()
#         init_ = init_easy(gain=nn.init.calculate_gain('relu'), bias=0)
#         # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
#         self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
#         self.relu1 = nn.ReLU()
#         self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
#         self.relu2 = nn.ReLU()
#         self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
#         self.relu3 = nn.ReLU()
#         # Flatten Here
#         self.linear = init_(nn.Linear(32 * 7 * 11, output_size))
#         self.relu4 = nn.ReLU()
#     def forward(self, inputs):
#         x = inputs.float()/255.
#         ## print("Shape of x INITIAL =", x.shape)
#         ## print("Dtype of x =", x.dtype)

#         x = self.relu1(self.conv1(x))
#         x = self.relu2(self.conv2(x))
#         x = self.relu3(self.conv3(x))
#         print(x.shape)
#         # flatten
#         ## print("Size of x BEFORE =", x.shape)
#         x = x.view(x.size(0), -1)
#         ## print("Size of x AFTER =", x.shape)
#         x = self.relu4(self.linear(x))
#         return x


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_size + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
