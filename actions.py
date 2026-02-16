# action_decoder.py

from typing import NamedTuple, Optional
import gymnasium as gym
import numpy as np
import torch
from scheduler import Scheduler
from gymnasium.vector import AsyncVectorEnv

class Action(NamedTuple):
    cmd:  str               # 'assign' or 'hold'
    core: Optional[int]     # core index for assign, else None
    slot: Optional[int]     # queue‐slot index for assign, else None

class ActionDecoder:
    def __init__(self, M: int, I: int, D_max: int):
        """
        M      : sliding‐window size (waiting‐queue length)
        I      : number of cores
        D_max  : maximum delay choices
        """
        self.M     = M
        self.I     = I
        self.D_max = D_max

        # gym action‐spaces
        self.assign_space = gym.spaces.Discrete(M * I + 1)
        self.delay_space  = gym.spaces.Discrete(D_max)

    # ---------- Encoding / Decoding ----------

    def decode_assign(self, a: int) :
        # print(f'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww{a}')
        # if not 0 <= a <= self.M * self.I:
        #     raise ValueError(f"Assign‐action {a} out of range [0..{self.M*self.I}]")
        if a == self.M * self.I:
            return Action('hold', None, None)

        core = a // self.M
        slot = a % self.M
        return Action('assign', core, slot)

    def encode_assign(self, core: int, slot: int) -> int:
        return core * self.M + slot

    def encode_hold(self) -> int:
        return self.M * self.I

    # ---------- Valid‐action masks ----------
    def valid_assign_mask(self, scheduler: Scheduler) -> torch.Tensor:
        """
        Returns a numpy boolean mask of length M*I+1 where True =
        legal assign or hold.  Always allows hold.
        """
        N = self.M * self.I + 1
        mask = np.zeros(N, dtype=bool)

        # check each assign‐action
        for a in range(self.M * self.I):
            cmd, core, slot = self.decode_assign(a)
            # slot must exist in the activeQueue, and check_assign must pass
            # print(scheduler.check_assign(core, slot))
            # if slot > len(scheduler.activeQueue) :
            #     print(f'slot is : {slot}, active : {len(scheduler.activeQueue)}')
            #     print("%%%%")
            # print(len(scheduler.activeQueue))
            # print(slot, len(scheduler.activeQueue))
            if (slot < len(scheduler.activeQueue)) and scheduler.check_assign(core, slot) :
                # print(slot, 'belong_to_valid_checker !')

                # print(self.M, 'belong_to_valid_checker as M  !')
                mask[a] = True

        # always allow hold
        mask[self.M * self.I] = True
        return  torch.tensor(mask, dtype=torch.bool)

    def max_delay_bound(self, scheduler, assign: int) -> int:
        """
        assign: int in [0 .. Q*C] where Q*C is HOLD
        returns: max_delay (>=0) as python int
        """
        # print(assign)
        if assign != self.M * self.I:
            core = assign // self.M
            slot = assign % self.M
        else :
            return 0
        md = int(scheduler.get_max_delay_time(core, slot))
        return max(0, md)


