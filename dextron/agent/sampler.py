import numpy as np
import warnings
from copy import deepcopy

from digideep.agent.sampler_common import get_memory_params # , Compose
from digideep.agent.ddpg.sampler import get_sample_memory
from digideep.utility.profiling import KeepTime


def append_common_keys(list_of_dicts):
    res = deepcopy(list_of_dicts[0])
    keys = list(res.keys())
    for index in range(1, len(list_of_dicts)):
        item = list_of_dicts[index]
        for key in keys:
            if key in item:
                # Append
                res[key] = np.append(res[key], item[key], axis=0)
            else:
                # Key does not exist in at least one of the items. So delete it from res as well.
                print("We had to delete [{}] key from base item.".format(key))
                del res[key]
    
    return res


# We need something to append all new data fields to the previous data field.
def multi_memory_sample(memory, infos):
    batch_size = 0
    list_of_buffers = []
    for m in memory:
        with KeepTime("info_deepcopy"):
            # 1. Make a copy of infos
            info = deepcopy(infos)
        with KeepTime("info_update"):
            # 2. Set the necessary changes in the corresponding info: Set the buffer_size for sampling.
            info["batch_size"] = infos["batch_size_dict"][m]
            batch_size += info["batch_size"]
            #### print("For {} we have bs = {}".format(m, info["batch_size"]))
        with KeepTime("get_memory_params"):
            # 3. Get the memory parameters of the specified key
            mem = get_memory_params(memory[m], info)
        with KeepTime("get_sample_memory"):
            # 4. Do the actual sampling from the memory
            buf = get_sample_memory(mem, info)
        if buf is None:
            # print("Not enough data in [{}].".format(m))
            return None
        
        # 5. Use demonstrator's actions not the agent's.
        # TODO: The correct thing is to use demonstrators actions. However, we want to test
        #       if it works with agents actions (this will definitely cause erroneous state
        #       trajectories which can effect learning system dynamics; because s' is not a
        #       subsequence of s by action a anymore.)
        ## if m == "replay":
        ##     # If we are in replaying mode, we want the demo's actions to be learnt.
        ##     buf["/agents/agent/actions"] = buf["/agents/demonstrator/actions"]

        # 6. Append the sampled buffers to a single buffer
        with KeepTime("append_buffer"):
            list_of_buffers += [buf]
    
    # print("{} vs. {}".format(np.mean(list_of_buffers[0]["/obs_with_key"]), np.mean(list_of_buffers[1]["/obs_with_key"])))

    with KeepTime("append_common_keys"):
        buffer = append_common_keys(list_of_buffers)
    
    with KeepTime("shuffle_inside"):
        # Shuffle inside the buffer:
        # NOTE: It does not matter most probably. Just in case ...
        p = np.random.permutation(batch_size)
        for key in buffer:
            buffer[key] = buffer[key][p, ...]

    # print(buffer.keys())
    # exit()

    # indices = (buffer["/observations/status/is_training"] == 0)
    # print(indices)
    # print(indices.shape)
    # exit()
    
    #### for k in buffer:
    ####     print("{:50s}: {} = {} + {}".format(k, buffer[k].shape, list_of_buffers[0][k].shape, list_of_buffers[1][k].shape))
    #### exit()
    return buffer



# def post_sampler(chunk, info):
#     # Use the demonstrator action when we are teaching!
#     if chunk:
#         assert chunk["/agents/agent/actions"].shape == chunk["/agents/demonstrator/actions"].shape, \
#             "The actions of interchangeable agents should have equal shape."
        
#         indices = (chunk["/observations/status/is_training"] == 0)
        
#         # There shouldn't be any indcies with is_training==0 because we are not saving the results of demo
#         # print(indices)
#         chunk["/agents/agent/actions"][indices] = chunk["/agents/demonstrator/actions"][indices]
#     return chunk

    