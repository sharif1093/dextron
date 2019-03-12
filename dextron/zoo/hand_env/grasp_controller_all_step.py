import numpy as np


class GraspController:
    """
    As the params: time_constant/duration of the motion.

    action set: {0, 1}
        - 0: Stay
        - 1: Close one step
    grasp type: 

                   Pre-grasp          Closed
                  
    Open  ________   power   _________ power
         |________   pinch   _________ pinch
         |________  tripod   _________ tripod

    Whenever we get a command we go to busy mode. Otherwise we are idle.
    We can get a command at the idle mode. But not at the busy mode.
    Agent should learn there is no undo.
    """
    def __init__(self, physics, **params):
        self.params = params
        
        MAX_STEP = 100

        self.jont_names = ["TCJ", "TPJ", "IPJ", "MPJ", "RPJ", "PPJ"]
        self.ctrl_names = ["A_thumb_C", "A_thumb_proximal", "A_index_proximal", "A_middle_proximal", "A_ring_proximal", "A_pinky_proximal"]
        self.ctrl_index = {key:physics.named.data.ctrl._convert_key(key) for key in self.ctrl_names}
        self.ctrl_cmnds = np.zeros(shape=(len(self.ctrl_names),))
        self.ctrl_targt = np.zeros(shape=(len(self.ctrl_names),))
        self.ctrl_range = {key:physics.named.model.actuator_ctrlrange[key] for key in self.ctrl_names}
        self.ctrl_minis = {key:self.ctrl_range[key][0] for key in self.ctrl_names}
        self.ctrl_maxis = {key:self.ctrl_range[key][1] for key in self.ctrl_names}
        self.ctrl_steps = {key:(self.ctrl_maxis[key]-self.ctrl_minis[key])/MAX_STEP for key in self.ctrl_names}
        
        self.state = {}
        self.state["node"] = "idle" # "pre-grasp", "open"

        # Put thumb in opposable position:
        # None-opposable thumb: 0 | Opposable thumb: 1.05
        self.ctrl_cmnds[self.ctrl_index["A_thumb_C"]] = 1.05

        self.ctrl_targt = [self.ctrl_maxis[key] for key in self.ctrl_names]

        self.last_action = 0

    def step(self, action, physics):
        # Change the "self.ctrl_cmnds" a bit with respect to "self.ctrl_targt"
        # My suggestion: Be obstacle aware! Watch joint positions and step while considering their values.
        # jont_posi = {key:physics.named.data.qpos[key] for key in self.jont_names}

        if action == 0:
            # Stay where you are.
            self.state["node"] = "idle"
            pass
        elif action == 1:
            # Flex
            flag = False
            for key in self.ctrl_names:
                if key == "A_thumb_C":
                    continue
                index = self.ctrl_index[key]
                if self.ctrl_cmnds[index] < self.ctrl_targt[index]:
                    self.ctrl_cmnds[index] += self.ctrl_steps[key]
                    flag = True
            
            self.state["node"] = "moving" if flag else "idle"
        # elif action == 2:
        #     # Extend
        #     for key in self.ctrl_names:
        #         if key == "A_thumb_C":
        #             continue
        #         index = self.ctrl_index[key]
        #         if self.ctrl_cmnds[index] > self.ctrl_minis[key]:
        #             self.ctrl_cmnds[index] -= self.ctrl_steps[key]
            

    def __call__(self, action, physics):

        if self.state["node"] == "moving":
            print("-", end='')
            sys.stdout.flush()
        elif self.state["node"] == "idle":
            print(action, end='')
            sys.stdout.flush()
            self.last_action = action
            self.state["node"] = "moving"
        
        self.step(self.last_action, physics)

        return self.ctrl_cmnds
