class MovingAverage:

    def __init__(self, momentum):
        self.m = momentum
        self.avg = 0.0

    def update(self, val):
        self.avg = val
        self.update = self._update
    
    def _update(self, val):
        self.avg = self.avg * self.m + val * (1 - self.m)

    def peek(self):
        return self.avg
