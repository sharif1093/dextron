#!/usr/bin/env python
# coding: utf-8

# # Read memory from .npz file
# * Read each subfile as a key in the memory dictionary.
# * Do some statistics on the data

# In[1]:


import numpy as np
np.random.seed(0)

loading = np.load("/sessions/session_20200914233251_objective_mccarthy/memsnapshot/checkpoint-1000/demo.npz")

buffer = {}
for key in loading.files:
    buffer[key] = loading[key]
    print(f"Key: {key} was added with dtype '{buffer[key].dtype}' and shape '{buffer[key].shape}'")


# In[2]:


#$

l = buffer["/infos/episode/l"]
l_where = np.where(l>0)

r = buffer["/infos/episode/r"]
r_where = np.where(r==r)

masks = buffer["/masks"]
masks_where = np.where(masks==0)

rewards = buffer["/rewards"]

assert len(masks_where[0])==len(l_where[0])
assert len(masks_where[0])==len(r_where[0])

n_episodes = len(masks_where[0])
n_workers = len(masks)

print(f"minimum length: {np.min(l[l_where])}; maximum length: {np.max(l[l_where])}; mean: {np.mean(l[l_where])}, std: {np.std(l[l_where])}")
print(f"minimum reward: {np.min(r[r_where])}; maximum reward: {np.max(r[r_where])}; mean: {np.mean(r[r_where])}, std: {np.std(r[r_where])}")


print(f"l:{l.shape}, r:{r.shape}, masks:{masks.shape}")

l_where_dict = {}
r_where_dict = {}
masks_where_dict = {}

for i in range(n_workers):
    l_where_dict[i] = np.where(l[i]>0)[0]
    r_where_dict[i] = np.where(r[i]==r[i])[0]
    masks_where_dict[i] = np.where(masks[i]==0)[0]


# In[ ]:





# # Several interesting problems
# 1. From states, predict actions! (*only for successful trajectories*)
# 2. From a sequence, predict the reward.
# 3. From a stack of states, predict the velocities.
# 4. From state and action, predict next state! (This is the environment model).
# 

# ## For problem 1: $a = \pi(s)$
# 
# > We only consider cases where: Rewards == 20

# In[ ]:





# In[3]:


# Remove index 0 of all datasets (which happens to be nan for /rewards for some reason.)
# Find the index of last completed trajectory in each worker. Call them ji's.
# Combine the first two indices by stacking [1:j1, 1:j2, 1:j3, ...] for all workers.
# Now all of the buffer arrays are the same size. The first index i for the stacked trajectories.

# Get the ends of trajectories
# masks_where = np.where(buffer["/masks"]==0)[:2]

# Set START=1 to remove the first nan elements in the buffer["/rewards"]
START=1
# Fixing episode lengths because of START=1
for i in range(n_workers):
    buffer["/infos/episode/l"][i,l_where_dict[i][0]] -= START


# Find last valid index of each worker.
masks_where_end_dict={}
for i in range(n_workers):
    masks_where_end_dict[i] = masks_where_dict[i][-1]

for i in range(1, 20):
    buffer["/masks"][i,START,0] = 0

# Now combine the first two axes by stacking them.
# Do it for all keys
stacks = {}
keys = list(buffer.keys())
for key in keys:
    shape = buffer[key].shape
    stack = [buffer[key][n,START:masks_where_end_dict[n]] for n in range(n_workers)]
    buffer[key] = np.concatenate(stack)
    print(f"Key: {key} change from '{shape}' to '{buffer[key].shape}'")

n_length = buffer["/masks"].shape[0]


# In[4]:


# Test time correctness
time_scale_offset = 0.5 # 1.0
time_scale_factor = 2.5 # 2.0
time_noise_factor = 0.8


noise = buffer["/infos/rand/time_noise_normal"] * time_noise_factor
T = buffer["/infos/rand/original_time"] * (time_scale_factor) + (time_scale_offset + noise)

assert np.linalg.norm(buffer["/infos/rand/randomized_time"] - T) < 1e-8


# In[ ]:





# In[5]:


#$

assert len(np.where(buffer["/masks"]==0)[0])+1 == n_episodes, "Number of episodes after concatenation should be the same as number of episodes before concatenation."
assert len(np.where(buffer["/infos/episode/r"]==buffer["/infos/episode/r"])[0]) == n_episodes
assert len(np.where(buffer["/infos/episode/l"]>0)[0]) == n_episodes


# In[ ]:





# In[6]:


#$

# After "combination", figure out the new whereabouts.
l_c = buffer["/infos/episode/l"]
l_where_c = np.where(l_c>0)[0]

r_c = buffer["/infos/episode/r"]
r_where_c = np.where(r_c==r_c)[0]
r20_where_c = np.where(r_c==20)[0]

masks_c = buffer["/masks"]
masks_where_c = np.where(masks_c==0)[0]
masks_where_c = np.concatenate([masks_where_c, [n_length]])

rewards_c = buffer["/rewards"]


# In[7]:


#$

# Check if all sets are the same.

assert len(masks_where_c)==len(l_where_c)
assert len(masks_where_c)==len(r_where_c)

assert len(set(masks_where_c-1) - set(l_where_c)) == 0
assert len(set(l_where_c) - set(masks_where_c-1)) == 0

assert len(set(masks_where_c-1) - set(r_where_c)) == 0
assert len(set(r_where_c) - set(masks_where_c-1)) == 0


# In[ ]:





# In[8]:


#$

# Another test here: r (sum of rewards at the end of each episode) should match stepwise rewards.
# r == sum(rewards[ending-20:ending])

for e in range(n_episodes):
    length = l_c[l_where_c[e]]
    reward = r_c[r_where_c[e]]
    ending = masks_where_c[e]
    
    reward_arr = buffer["/rewards"][ending-min(20,length-1):ending]
    
    assert sum(reward_arr)==reward, f"Reward discripancy for episode={e}, length={length}, reward={reward}, index={ending}"


# In[ ]:





# ### Create another column with trajectory index

# In[9]:


assert n_episodes == len(l_where_c)

buffer["/infos/episode/i"] = np.empty_like(buffer["/infos/episode/r"], dtype=np.int)
buffer["/infos/episode/timestep"] = np.empty_like(buffer["/infos/episode/r"], dtype=np.int)

j = 0
for i in range(n_episodes):
    timestep = 0
    while j <= l_where_c[i]:
        buffer["/infos/episode/i"][j] = i
        buffer["/infos/episode/timestep"][j] = timestep
        timestep+=1
        j+=1


# In[10]:


# Test: Shouldn't l_c[l_where_c[1]]==l_where_c[1]-l_where_c[0]
arrr=0
for i in range(1,n_episodes):
    assert l_c[l_where_c[i]] == (l_where_c[i]-l_where_c[i-1]), f"In {i}, {l_c[l_where_c[i]]} is not equal to {l_where_c[i]-l_where_c[i-1]}."
    # arrr += np.abs(l_c[l_where_c[i]] - (l_where_c[i]-l_where_c[i-1]))


# ### Create time to reach column

# In[11]:


# Get start_time, final_time, reach_time from the .json files.
# Then convert reach time to timesteps, and store that along other keys.


# First create a dictionary for all files
import json, glob, os

meta_dict = {}
files = sorted(glob.glob("../extracts/*.json"))
for file in files:
    name = os.path.splitext(os.path.split(file)[1])[0]
    
    with open(f'../extracts/{name}.json') as f:
        lines = f.readlines()
        start_time = json.loads(lines[0])["t"]
        total_time = json.loads(lines[-1])["t"] - start_time
    with open(f'../extracts/meta/{name}_meta.json') as f:
        reach_time = dict(json.load(f))['reached'] - start_time
    
    meta_dict[name] = {"start_time":start_time, "reach_time":reach_time, "total_time":total_time}


# Iterate on each entry, add "reach timestep" to 
CONTROLLER_STEP = 0.02
noise = buffer["/infos/rand/time_noise_normal"] * time_noise_factor
buffer["/infos/episode/reach_timestep"] = np.empty((n_length,))


for i in range(n_length):
    reach_time = meta_dict[buffer["/infos/rand/filename"][i]]["reach_time"]
    T = reach_time*time_scale_factor + (time_scale_offset+noise[i])
    buffer["/infos/episode/reach_timestep"][i] = int(T / CONTROLLER_STEP)


# In[ ]:





# In[12]:


# buffer["/infos/episode/l"]


# ## Dataset creation
# 
# Identify which trajectories where successful, i.e. they had r=20 at the end. Then, use data from those trajectories 

# In[13]:


# Create a list of acceptable indices to sample from:

indices = []
for i in range(len(r20_where_c)):
    e = r20_where_c[i]
    s = e - (l_c[e]-1)
    assert r_c[e] == 20
    
    indices += list(range(s+1,e+2))

assert sum(l_c[r20_where_c]) == len(indices), "Sum of all valid lengths (with r==20) must be equal to the number of all valid indices."

# BUG: We remove very last element since that element probably does not exist.
indices = indices[:-1]


# In[ ]:





# In[14]:


#@

# Split data to training, testing, and validating data.
np.random.seed(0)
np.random.shuffle(indices)
N = len(indices)

N80 = int(0.80 * N)

training, test = indices[:N80], indices[N80:]


# In[ ]:





# In[15]:


# # states
# x = 

# # actions
# y = 

# # Try to find model f
# # Can be BN, MLP, BN, MLP, BN, ...
# y = f(x)


# In[16]:


import torch

# TODO: Sets the number of OpenMP threads used for parallelizing CPU operations
# torch.set_num_threads(1)
        
## GPU
cuda_available = torch.cuda.is_available()
if cuda_available: # and use_gpu:
    print("GPU available. Using 1 GPU.")
    device = torch.device("cuda")
    # dtype = torch.cuda.FloatTensor
    # dtypelong = torch.cuda.LongTensor
else:
    print("Using CPUs.")
    device = torch.device("cpu")
    # dtype = torch.FloatTensor
    # dtypelong = torch.LongTensor

# model = ?


# In[17]:


# import torch.nn as nn
# # import torch.distributions as distributions

# class ModelClass(nn.Module):
#     def __init__(self, state_size, hidden_size, action_size, action_scale, init_w=3e-3):
#         super(ModelClass, self).__init__()
#        
#         self.linear1 = nn.Linear(state_size,  hidden_size)
#         self.bn1     = nn.BatchNorm1d(num_features=hidden_size)
#         self.linear2_1 = nn.Linear(hidden_size, hidden_size)
#         self.bn2_1     = nn.BatchNorm1d(num_features=hidden_size)
#         self.linear2_2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2_2     = nn.BatchNorm1d(num_features=hidden_size)
#         self.linear2_3 = nn.Linear(hidden_size, hidden_size)
#         self.bn2_3     = nn.BatchNorm1d(num_features=hidden_size)
#         self.linear2_4 = nn.Linear(hidden_size, hidden_size)
#         self.bn2_4     = nn.BatchNorm1d(num_features=hidden_size)
#         self.linear3 = nn.Linear(hidden_size, action_size)
#        
#         self.linear3.weight.data.uniform_(-init_w, init_w)
#         self.linear3.bias.data.uniform_(-init_w, init_w)
#        
#         self.action_scale = action_scale
#        
#     def forward(self, state):
#         # x = torch.cat([state, action], 1)
#         x = state
#         x = torch.relu(self.bn1(self.linear1(x)))
#         x = torch.relu(self.bn2_1(self.linear2_1(x)))
#         x = torch.relu(self.bn2_2(self.linear2_2(x)))
#         x = torch.relu(self.bn2_3(self.linear2_3(x)))
#         x = torch.relu(self.bn2_4(self.linear2_4(x)))
#         x = torch.tanh(self.linear3(x)) * self.action_scale
#         return x


# In[20]:


import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, action_scale, init_w=3e-3):
        super(ModelClass, self).__init__()
        
        self.linear1 = nn.Linear(state_size,  hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
        self.action_scale = action_scale
        
    def forward(self, state):
        # x = torch.cat([state, action], 1)
        x = state
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # x = torch.tanh(self.linear3(x)) * self.action_scale
        x = self.linear3(x)
        return x


# In[26]:


# Run this block again to reset the model and optimizer.

from torch.optim import Adam
import torch.nn as nn

model = ModelClass(state_size=48, hidden_size=256, action_size=1, action_scale=10)
# # Multi-GPU
# if torch.cuda.device_count() >= 1:
#     gpu_count = torch.cuda.device_count()
#     model = nn.DataParallel(model)
# else:
#     gpu_count = 0
model = model.to(device)

optimizer = Adam(model.parameters(),
                 lr=0.0003,
                 betas=(0.9, 0.999),
                 eps=1e-08)
lossfn = nn.MSELoss()


# In[ ]:


from tqdm.auto import tqdm, trange


n_epochs = 100
batch_size = 32
best_loss = np.infty

for e in trange(n_epochs, desc="epoch"):
    # Train
    model.train() # model.training = true
    np.random.shuffle(training)
    n_batches = len(training) // batch_size
    training_loss = 0
    for b in trange(n_batches, desc="train", leave=False):
        # Select data for that batch
        batch_indices = training[b*batch_size:(b+1)*batch_size]
        
        position_np = buffer["/observations/agent/position"][batch_indices]
        velocity_np = buffer["/observations/agent/velocity"][batch_indices]
        
        timestep_to_reach = buffer["/infos/episode/reach_timestep"][batch_indices] - buffer["/infos/episode/timestep"][batch_indices]
        # timestep_to_reach = np.clip(timestep_to_reach, 0, np.infty)
        # timestep = buffer["/infos/episode/timestep"][batch_indices]
        # actions_np  = buffer["/agents/demonstrator/actions"][batch_indices]
        
        states  = torch.from_numpy(np.concatenate([position_np,velocity_np], axis=1)).to(device).float()
        # states  = torch.from_numpy(np.concatenate([position_np,velocity_np,timestep_to_reach.reshape(-1,1)], axis=1)).to(device).float()
        # action_desired = torch.from_numpy(actions_np).to(device).float()
        timestep_to_reach_desired = torch.from_numpy(timestep_to_reach.reshape(-1,1)).to(device).float()
        
        # Forward
        # action_model = model(states)
        timestep_to_reach_model = model(states)
        

        # loss = lossfn(action_model, action_desired)
        loss = lossfn(timestep_to_reach_model, timestep_to_reach_desired)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        training_loss += loss.item()
        
    training_loss = training_loss / n_batches
    
    
    # Test
    model.eval() # model.training = true
    n_batches = len(test) // batch_size
    test_loss = 0
    for b in trange(n_batches, desc="test", leave=False):
        # Select data for that batch
        batch_indices = test[b*batch_size:(b+1)*batch_size]
        
        position_np = buffer["/observations/agent/position"][batch_indices]
        velocity_np = buffer["/observations/agent/velocity"][batch_indices]
        
        timestep_to_reach = buffer["/infos/episode/reach_timestep"][batch_indices] - buffer["/infos/episode/timestep"][batch_indices]
        # timestep_to_reach = np.clip(timestep_to_reach, 0, np.infty)
        # timestep = buffer["/infos/episode/timestep"][batch_indices]
        # actions_np  = buffer["/agents/demonstrator/actions"][batch_indices]
        
        states  = torch.from_numpy(np.concatenate([position_np,velocity_np], axis=1)).to(device).float()
        # states  = torch.from_numpy(np.concatenate([position_np,velocity_np,timestep_to_reach.reshape(-1,1)], axis=1)).to(device).float()
        # action_desired = torch.from_numpy(actions_np).to(device).float()
        timestep_to_reach_desired = torch.from_numpy(timestep_to_reach.reshape(-1,1)).to(device).float()
        
        # Forward
        # action_model = model(states)
        timestep_to_reach_model = model(states)

        # loss = lossfn(action_model, action_desired)
        loss = lossfn(timestep_to_reach_model, timestep_to_reach_desired)
        
        # Accumulate loss
        test_loss += loss.item()
    
    test_loss = test_loss / n_batches
    
    print(f"Epoch {e}: mean training loss: {training_loss}, mean test loss: {test_loss}.")
    
    if test_loss < best_loss:
        torch.save(model.state_dict(), f"backups/model_epoch_{e}_{test_loss:5.4f}.pt")
        best_loss = test_loss
    
    
    
        
    # Compute test error
    # Report
    # Early stopping?
    
    
    
        
        
        


# In[22]:


timestep_to_reach


# In[23]:


np.clip(timestep_to_reach, 0, np.infty)


# In[ ]:





# In[ ]:





# In[28]:


states.shape


# In[ ]:





# In[ ]:





# In[14]:


best_loss = np.infty
print(best_loss*3)


# In[30]:


abc = 34341.123445676767887


# In[31]:





# In[84]:


torch.save(model.state_dict(), "model_epoch_1.pt")


# In[59]:


model = ModelClass(state_size=48, hidden_size=256, action_size=1, action_scale=10)
model = model.to(device)

# state_dict = torch.load("backups/model_epoch_1.pt")


# In[41]:


action_desired


# In[61]:


torch.mean((model(states) - action_desired)**2)


# In[60]:


list(zip(model(states).detach().cpu().numpy(),action_desired.cpu().numpy()))


# In[54]:


action_desired.cpu().numpy()


# In[35]:


position = buffer["/observations/agent/position"][batch_indices]
velocity = buffer["/observations/agent/velocity"][batch_indices]
actions  = buffer["/agents/demonstrator/actions"][batch_indices]

masks    = buffer["/masks"][batch_indices]
rewards  = buffer["/rewards"][batch_indices]
# filename = buffer["/infos/rand/filename"][batch_indices]
# timestep = buffer["/agents/demonstrator/hidden_state/time_step"][batch_indices]

# initial_closure = buffer["/observations/parameters/initial_closure"][batch_indices]
# controller_thre = buffer["/observations/parameters/controller_thre"][batch_indices]
# controller_gain = buffer["/observations/parameters/controller_gain"][batch_indices]

# hand_closure = buffer["/observations/demonstrator/hand_closure"][batch_indices]
# rel_obj_hand = buffer["/observations/agent/rel_obj_hand"][batch_indices]
# rel_obj_hand_dist = buffer["/observations/agent/rel_obj_hand_dist"][batch_indices]

# distance2 = buffer["/observations/agent/distance2"][batch_indices]
# closure = buffer["/observations/agent/closure"][batch_indices]
# distance = buffer["/observations/demonstrator/distance"][batch_indices]


# In[36]:


print(position.shape)
print(velocity.shape)
print(actions.shape)
print(filename)
print(timestep.reshape(-1))
print(masks.reshape(-1))
print(rewards.reshape(-1))


# In[39]:


position.shape


# In[41]:


velocity.shape


# In[43]:


np.concatenate([position,velocity], axis=1).shape


# In[ ]:





# In[ ]:





# In[ ]:


# Load data. Create a dataloader/dataset interface: Not needed! All data is in the memory!

# Preprocess, e.g. do data augmentation.
#   In early stages that is not usually required. 
# Visualize the samples you do.
# Split data into train/test
# Train model. 
# Once trained, test and try it! Especially on unseen data.


# In[ ]:





# In[ ]:


# Load data
# Preprocess data
# Pre visualizations

# Assuming optimizer has two groups.
lambda1 = lambda epoch: epoch \\ 30
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
for epoch in range(100):
    scheduler.step()
    
    ### Train
    model.train() # model.training = true

    # For all data:
    #   data, label = ...
    #   inputs = inputs.to(device)
    #   labels = labels.to(device)
    #   TRACKING = ON
    #   forward: y = model(x)
    #   Compute Loss
    #   optimizer.zero_grad()
    #   loss.backward()
    #   Update with learning_rates: optimizer.step()
    #   ---
    #   Compute total loss and number of correct guesses
    # ------------------------
    # Report loss and accuracy
    
    ### Validate
    model.eval() # model.training = false
    # For all data:
    #   data, label = ...
    #   inputs = inputs.to(device)
    #   labels = labels.to(device)
    #   TRACKING = OFF,
    #   forward: y = model(x)
    #   Compute Loss
    #   ---
    #   Compute total loss and number of correct guesses
    # ------------------------
    # Report loss and accuracy
    # Keep track of the best epoch so far
    #      best_model_state = copy.deepcopy(model.state_dict())

# Post-processing


# In[ ]:





# In[ ]:





# In[ ]:





# ## All memory keys
# 
# * `/observations/agent/position`, `float64`, (19997984, 25)
# * `/observations/agent/velocity`, `float64`, (19997984, 23)
# * `/observations/agent/rel_obj_hand`, `float64`, (19997984, 1, 3)
# * `/observations/agent/rel_obj_hand_dist`, `float64`, (19997984,)
# * `/observations/agent/distance2`, `float64`, (19997984,)
# 
# * `/observations/agent/closure`, `float64`, (19997984,)
# * `/observations/demonstrator/distance`, `float32`, (19997984,)
# * `/observations/demonstrator/hand_closure`, `float32`, (19997984,)
# * `/observations/status/is_training`, `uint8, (19997984,)
# * `/observations/parameters/initial_closure`, `float64`, (19997984,)
# 
# * `/observations/parameters/controller_gain`, `float64`, (19997984,)
# * `/observations/parameters/controller_thre`, `float64`, (19997984,)
# * `/observations/parameters/real_trajectory`, `uint8`, (19997984,)
# * `/masks`, `float32`, (19997984, 1)
# * `/agents/demonstrator/actions`, `float32`, (19997984, 1)
# 
# * `/agents/demonstrator/hidden_state/time_step`, `float32`, (19997984, 1)
# * `/agents/demonstrator/hidden_state/initial_distance`, `float32`, (19997984, 1)
# * `/agents/demonstrator/hidden_state/controller_gain`, `float32`, (19997984, 1)
# * `/agents/demonstrator/hidden_state/controller_thre`, `float32`, (19997984, 1)
# * `/rewards`, `float64`, (19997984, 1)
# 
# * `/infos/rand/filename`, `<U8`, (19997984,)
# * `/infos/rand/time_noise_normal`, `float64`, (19997984,)
# * `/infos/rand/offset_noise_2d`, `float64`, (19997984, 3)
# * `/infos/rand/original_time`, `float64`, (19997984,)
# * `/infos/rand/randomized_time`, `float64`, (19997984,)
# 
# * `/infos/episode/r`, `float64`, (19997984,)
# * `/infos/episode/l`, `int64`, (19997984,)
# * `/infos/episode/t`, `float64`, (19997984,)

# In[ ]:





# In[ ]:




