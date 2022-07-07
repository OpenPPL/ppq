from math import pow

class EMARecorder():
    """Exponential Moving Average(EMA) with bias correction."""
    def __init__(self, beta: float = 0.98):
        self.beta  = beta
        self.t     = 0
        self.value = 0

    def push(self, value: float):
        self.value = (1.0 - self.beta) * value + self.beta * self.value
        self.t += 1

    def pop(self) -> float:
        if self.t == 0: return 0
        return self.value / (1 - pow(self.beta, self.t))