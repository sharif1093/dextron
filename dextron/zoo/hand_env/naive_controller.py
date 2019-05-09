import numpy as np


class NaiveController:
    """
    Tha NaiveController is a model-based controller which has access to all underlying
    parameters/variables that RL does not have access to necessarily. This controller
    shall be used as a "naive" teacher.
    """
    def __init__(self, action_spec):
        self.action_spec = action_spec
        self.states = {}
        # Currently the action spec is only one bounded region of parameter t: 0 <= t <= 1
        self._delayed_init = False
    
    def delayed_init(self, physics):
        self.states["initial_distance"] = physics.get_distance_in_xy_plane()
        # print("Initial distance:", self.states["initial_distance"])
        
    def step(self, physics): # State feedbacks come from "physics"
        # Lazy initialization of the controller
        if not self._delayed_init:
            self.delayed_init(physics)
            self._delayed_init = True

        hand_closure = physics.get_hand_closure()
        dist = physics.get_distance_in_xy_plane()
        dist_normalized = dist / self.states["initial_distance"]

        # print("dist =", dist)
        
        
        DISTANCE_CRITICAL = 0.15
        OPEN_HAND_CLOSURE = 0.2
        
        CLOSE_HAND_CLOSURE = 0.8

        GRASPER_GAIN = 2

        ## Control Law
        if dist_normalized >= DISTANCE_CRITICAL:
            # print("Now openning ...")
            cmd = GRASPER_GAIN * (OPEN_HAND_CLOSURE - hand_closure)
        elif dist_normalized < DISTANCE_CRITICAL:
            # print("Now closing ...")
            cmd = GRASPER_GAIN * (CLOSE_HAND_CLOSURE - hand_closure)
        
        
        actions = np.array([cmd], dtype=np.float)
        # Check if the action values are within the defined structure, then return.
        # self.action_spec["agent"].validate(actions)
        return actions
        


