import numpy as np
import warnings

def post_sampler(chunk, info):
    # Use the demonstrator action when we are teaching!
    if chunk:
        assert chunk["/agents/agent/actions"].shape == chunk["/agents/demonstrator/actions"].shape, \
            "The actions of interchangeable agents should have equal shape."
        
        indices = (chunk["/observations/status/is_training"] == 0)
        chunk["/agents/agent/actions"][indices] = chunk["/agents/demonstrator/actions"][indices]
    return chunk

    