import numpy as np

_THUMB_NON_OPPOSABLE = 0
_THUMB_OPPOSABLE = 1.05

class GraspControllerAllStep:
    """
    ASSUMES position controlled actuators beneath.
    ONLY supports two actions: (stay/close)
    IF used with velocity based actuators may result in unexpected behavior.

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
        self.ctrl_cmnds[self.ctrl_index["A_thumb_C"]] = _THUMB_OPPOSABLE

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







class GraspControllerAllPosition:
    """
    ASSUMES position controlled actuators beneath.
    ONLY assigns actions directly to the actuators.
    IF used with velocity controlled actuators may still work.

    As the params: time_constant/duration of the motion.
    """
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
        self.ctrl_cmnds[self.ctrl_index["A_thumb_C"]] = _THUMB_OPPOSABLE

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









class GraspControllerAllVelocity:
    """
    ASSUMES velocity controlled actuators beneath.
    """

    def __init__(self, physics, **params):
        self.params = params

        # I keep these lines because joint range is used in step().
        self.jont_names = ["TCJ", "TPJ", "IPJ", "MPJ", "RPJ", "PPJ"]
        self.jont_range = {key:physics.named.model.jnt_range[key] for key in self.jont_names}
        self.jont_direc = np.array([1, -1, 1, 1, 1, 1], dtype=np.float) # Can only be 1 or -1
        self.jont_minis = np.array([self.jont_range[key][0 if self.jont_direc[index]>0 else 1]*self.jont_direc[index] for index,key in enumerate(self.jont_names)], dtype=np.float)
        self.jont_maxis = np.array([self.jont_range[key][1 if self.jont_direc[index]>0 else 0]*self.jont_direc[index] for index,key in enumerate(self.jont_names)], dtype=np.float)

        self.max_length = np.max(self.jont_maxis - self.jont_minis)

        self.ctrl_names = ["A_thumb_C", "A_thumb_proximal", "A_index_proximal", "A_middle_proximal", "A_ring_proximal", "A_pinky_proximal"]
        self.ctrl_index = {key:physics.named.data.ctrl._convert_key(key) for key in self.ctrl_names}
        self.ctrl_cmnds = np.zeros(shape=(len(self.ctrl_names),))
        # self.ctrl_targt = np.zeros(shape=(len(self.ctrl_names),))
        self.ctrl_range = {key:physics.named.model.actuator_ctrlrange[key] for key in self.ctrl_names}
        self.ctrl_minis = {key:self.ctrl_range[key][0] for key in self.ctrl_names}
        self.ctrl_maxis = {key:self.ctrl_range[key][1] for key in self.ctrl_names}

        # Put thumb in opposable position:
        self.ctrl_cmnds[self.ctrl_index["A_thumb_C"]] = _THUMB_OPPOSABLE

        # self.ctrl_targt = [self.ctrl_maxis[key] for key in self.ctrl_names]


    def step(self, action, physics):
        t = action
        # t = 1

        # t += np.random.rand() * 1
        # 1: Fully closed
        # 0: Fully open
        flag = False
        for key in self.ctrl_names:
            if key == "A_thumb_C":
                continue
            index = self.ctrl_index[key]
            
            length = self.jont_maxis[index] - self.jont_minis[index]
            # self.ctrl_cmnds[index] = self.ctrl_minis[key] + (self.ctrl_maxis[key] - self.ctrl_minis[key]) * t
            # Normalize
            # self.ctrl_maxis[key] - self.ctrl_minis[key]
            # NOTE: Be careful about super large control commands.
            # NOTE: Be careful about "saturation" of commands with "ctrlrange".
            self.ctrl_cmnds[index] = (t / self.max_length) * length
            # TODO: We can do the saturation here ourself, and then normalize saturated commands afterwards to save proportionality.

            # if self.ctrl_cmnds[index] > self.ctrl_maxis[key]:
            #     print(key, "control command is greater than actuator range.")
            # elif self.ctrl_cmnds[index] < self.ctrl_minis[key]:
            #     print(key, "control command is smaller than actuator range.")
            
    def __call__(self, action, physics):
        self.step(action, physics)
        return self.ctrl_cmnds

