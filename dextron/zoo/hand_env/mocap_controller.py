class MocapController:
    def __init__(self, generator):
        """
        `generator` should be an array.
        """
        self.generator = generator
        self.termination = False
        self.index = 0

        self.mocap_pos = None
        self.mocap_quat = None

    def step(self, physics):
        # Setting the mocap bodies to move:
        try:
            # Output is expected to be `t, x, q`
            _, self.mocap_pos, self.mocap_quat = next(self.generator[self.index])
            if not self.mocap_pos is None:
                physics.named.data.mocap_pos["mocap"] = self.mocap_pos
            if not self.mocap_quat is None:
                physics.named.data.mocap_quat["mocap"] = self.mocap_quat
        except StopIteration:
            # NOTE: This part handles piece-wise trajectory generators.
            #       We have a list of generators, whenever we reach end
            #       of a generator, we switch to the next generator.
            if self.index < len(self.generator)-1:
                self.index += 1
            else:
                self.termination = True

