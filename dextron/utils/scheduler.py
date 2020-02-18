class Scheduler:
    def __init__(self, initial, interval, decay):
        self.__value = initial
        self.interval = interval
        self.decay = decay
        self.count = 0
    def update(self):
        self.count += 1
        if self.count % self.interval == 0:
            self.__value *= self.decay
    @property
    def value(self):
        return self.__value
