import numpy as np


class GraspController:
    def __init__(self, physics, **params):
        self.params = params
        
        MAX_STEP = 100

        # self.jont_names = ["TCJ", "TPJ", "IPJ", "MPJ", "RPJ", "PPJ"]
        self.ctrl_names = ["A_thumb_C", "A_thumb_proximal", "A_index_proximal", "A_middle_proximal", "A_ring_proximal", "A_pinky_proximal"]
        self.ctrl_index = {key:physics.named.data.ctrl._convert_key(key) for key in self.ctrl_names}
        self.ctrl_cmnds = np.zeros(shape=(len(self.ctrl_names),))
        # self.ctrl_targt = np.zeros(shape=(len(self.ctrl_names),))
        self.ctrl_range = {key:physics.named.model.actuator_ctrlrange[key] for key in self.ctrl_names}
        self.ctrl_minis = {key:self.ctrl_range[key][0] for key in self.ctrl_names}
        self.ctrl_maxis = {key:self.ctrl_range[key][1] for key in self.ctrl_names}
        # self.ctrl_steps = {key:(self.ctrl_maxis[key]-self.ctrl_minis[key])/MAX_STEP for key in self.ctrl_names}
        
        # self.state = {}
        # self.state["node"] = "idle" # "pre-grasp", "open"

        # Put thumb in opposable position:
        # None-opposable thumb: 0 | Opposable thumb: 1.05
        self.ctrl_cmnds[self.ctrl_index["A_thumb_C"]] = 1.05
        # 0:    lateral non-opposed
        # 1.05: opposed

        # self.ctrl_targt = [self.ctrl_maxis[key] for key in self.ctrl_names]

        # self.last_action = 0

    def step(self, action, physics):
        t = action
        # t += np.random.rand() * 1
        # 1: Fully closed
        # 0: Fully open
        flag = False
        for key in self.ctrl_names:
            if key == "A_thumb_C":
                continue
            index = self.ctrl_index[key]
            
            self.ctrl_cmnds[index] = self.ctrl_minis[key] + (self.ctrl_maxis[key] - self.ctrl_minis[key]) * t
            

    def __call__(self, action, physics):
        self.step(action, physics)
        return self.ctrl_cmnds

