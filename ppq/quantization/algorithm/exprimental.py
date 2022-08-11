import random
from typing import List

import numpy as np
import torch
from ppq.core import TensorQuantizationConfig
from ppq.executor import TorchQuantizeDelegator
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction
from ppq.utils.ema import EMARecorder


class BanditDelegator(TorchQuantizeDelegator):
    """带有多臂赌博机的量化代理，从 ppq 0.6.2 版本后，我们引入 多臂赌博机算法训练 scale 与 offset。在未来我们可能还会引入其他
    类似的算法，例如UCB，马尔可夫蒙特卡洛估计等。

    引入这些算法的原因是我们注意到 scale 与 offset 的导数非常不靠谱
    为此我们引入简单的强化学习，直接估计P(r | scale=s, context)
    即再给定上下文 context 的情况下，选取当前 scale 为 s，获利的概率

    Quantization with multi-arm bandit.

    Multi-arm bandits are introduced since PPQ 0.6.2 for training
        quantization scale and offset.
    """
    def __init__(self,  arms: List[float], config: TensorQuantizationConfig) -> None:
        if len(arms) < 2: raise ValueError('Can not initialize bandit with less than 2 arms.')
        self.e = 0.1
        self.arms = arms
        self.num_of_arms = len(arms)
        self.rewards = [EMARecorder() for _ in range(self.num_of_arms)]
        self.rewards[0].push(1)
        self.last_selected = 0
        self.reference = config.scale.clone()
        self.config = config
        self.decay = 0.99

    def roll(self) -> int:
        if random.random() > self.e: selected = random.randint(0, len(self.arms) - 1)
        else: selected = np.argmax([ema.pop() for ema in self.rewards])
        self.last_selected = selected
        return selected

    def mark(self, rewards: float):
        self.rewards[self.last_selected].push(rewards)

    def finalize(self) -> bool:
        self.config.scale = self.reference * self.arms[np.argmax([ema.pop() for ema in self.rewards])]

    def withdraw(self):
        self.config.scale = self.reference

    def __call__(self, tensor: torch.Tensor,
                 config: TensorQuantizationConfig) -> torch.Tensor:
        config.scale = self.reference * self.arms[self.roll()]
        return PPQLinearQuantFunction(tensor, config)