# import random
# import math
from typing import List, Callable, Union, Dict
import numpy as np
class Core:
    """
    Represents a compute core with specific multiplier capabilities,
    **and** a known average power draw when active.
    """
    def __init__(self,
                 exact_int: bool   = False,
                 approx_int: bool  = False,
                 exact_fp: bool    = False,
                 approx_fp: bool   = False,
                 avg_power: float  = 1.0):
        """
        avg_power : average power consumption when executing (in watts
                    or energy‐per‐tick units—just be consistent).
        """
        self.exact_int  = exact_int
        self.approx_int = approx_int
        self.exact_fp   = exact_fp
        self.approx_fp  = approx_fp

        # new: average power (energy per unit runtime)
        self.avg_power = avg_power

    def capabilities_bits(self) -> List[float]:
        """
        Returns a fixed-length binary vector describing this core's multiplier units.
        Order: [exact_int, approx_int, exact_fp, approx_fp]
        """
        return [
            float(self.exact_int),
            float(self.approx_int),
            float(self.exact_fp),
            float(self.approx_fp),
        ]

    def __repr__(self):
        caps = "".join([
            "Ei" if self.exact_int else "",
            "Ai" if self.approx_int else "",
            "Ef" if self.exact_fp else "",
            "Af" if self.approx_fp else "",
        ]) or "Idle"
        return f"<Core {caps}, P={self.avg_power}>"

class Task:
    """
    Represents a computational job with timing, resource needs, and energy requirements.
    """

    def __init__(self,
                 task_name: str,
                 arrival_time: int,
                 deadline: int,
                 req_exact_int: bool,
                 req_approx_int: bool,
                 req_exact_fp: bool,
                 req_approx_fp: bool,
                 rt_est: List[Union[float, int]],  # estimated runtimes per core
                 WCEC: List[float]):                # worst-case energy per core
        self.task_name = task_name
        self.finished_on_time = False
        self.task_tag = [self.task_name, arrival_time]
        # arrival / deadline
        self.arrival_time = arrival_time
        self.deadline     = deadline
        self.start_time = 0
        self.finish_time = 0
        # multiplier requirements
        self.req_exact_int  = req_exact_int
        self.req_approx_int = req_approx_int
        self.req_exact_fp   = req_exact_fp
        self.req_approx_fp  = req_approx_fp

        # per-core performance/energy
        self.rt_est = rt_est
        self.WCEC   = WCEC

        # **new**: initialize waiting / slack
        self.waiting_time = 0.0  
        self.slack_time   = 0.0
        self.back_log_score = max (self.rt_est) * max (self.WCEC)

    def feature_vector(self, current_time: int) -> List[float]:
        """
        Builds the task features for state input, *and* updates
        self.waiting_time / self.slack_time so you can read them later.
        """
        # 1) recompute & store
        self.waiting_time = float(current_time - self.arrival_time)
        self.slack_time   = float(self.deadline - current_time)

        # 2) build the rest of the vector
        bits = [
            float(self.req_exact_int),
            float(self.req_approx_int),
            float(self.req_exact_fp),
            float(self.req_approx_fp),
        ]
        return [
            self.waiting_time,
            self.slack_time ,
            *bits,
            *[float(r) for r in self.rt_est],
            *self.WCEC,
        ]

    @staticmethod
    def feature_length(num_cores: int) -> int:
        return 2 + 4 + 2 * num_cores


class ChargeRateProcess:
    """
    Abstract base for charge rate generators.  Subclasses define get_rate(t).
    """
    def get_rate(self, t: int) -> float:
        raise NotImplementedError

class ConstantChargeRate(ChargeRateProcess):
    """
    Simple constant charge rate.
    """
    def __init__(self, rate: float):
        self.rate = rate
    def get_rate(self, t: int) -> float:
        return self.rate

class SolarChargeRate(ChargeRateProcess):
    """
    Charge‐rate = α · A · irr(slot), where slot is one of
    96 fifteen‐minute intervals per day, keyed 0..95.
    """

    def __init__(self,
                 alpha:       float,
                 area:        float,
                 irr_profile: Dict[int, float],
                 interval_ms: int = 15 * 60 * 1000):
        """
        alpha        : PV conversion efficiency (0..1)
        area         : effective area (m²)
        irr_profile  : dict mapping slot index 0..N-1 → irradiance (W/m²)
                       where N = (24*60*60*1000) // interval_ms
        interval_ms  : length of each slot in ms (default 15min)
        """
        self.alpha       = alpha
        self.area        = area
        self.interval_ms = interval_ms

        # Compute how many intervals fit into one day
        day_ms = 24 * 60 * 60 * 1000
        self.num_slots = day_ms // interval_ms
        # print(self.num_slots)
        if day_ms % interval_ms != 0:
            raise ValueError(f"interval_ms must divide 24h in ms exactly")

        # Check profile completeness
        missing = set(range(self.num_slots)) - set(irr_profile)
        if missing:
            raise ValueError(f"irr_profile missing slots: {sorted(missing)}")
        self.irr_profile = irr_profile

    def get_rate(self, t: int) -> float:
        """
        t: global time in milliseconds.
        Returns energy harvested in this tick (units: W/m²·m² = W).
        """
        # Figure out which 15-minute slot we’re in
        slot = ((t // self.interval_ms) % self.num_slots) - 1
        # x = ((t // self.interval_ms) % self.num_slots)
        # print(t)
        # print (slot)
        irr  = self.irr_profile[slot]
        # print(irr)
        return self.alpha * self.area * irr

    def forecast(self, t: int, horizon: int) -> List[float]:
        """
        Returns the next `horizon` slot rates, starting at t.
        """
        out = []
        base_slot = (t // self.interval_ms) % self.num_slots
        for i in range(horizon):
            slot = (base_slot + i) % self.num_slots
            # print(slot)
            out.append(self.alpha * self.area * self.irr_profile[slot])
        # print (out)
        return out
class Battery:
    """
    Models a shared battery driven by a ChargeRateProcess.
    Handles variable‐duration charging correctly.
    """
    def __init__(self,
                 capacity: float,
                 rate_process: ChargeRateProcess,
                 initial_charge=None,
                 begining_time: float = 0.0):
        self.capacity     = float(capacity)
        self.rate_process = rate_process
        # convert hours → milliseconds
        self._start_offset_ms = int(begining_time)
        if initial_charge is None:
            initial_charge = self.capacity
        self.initial = max(0.0, min(float(initial_charge), self.capacity))
        self.current = self.initial
        # print(self.current, self.initial)
    def get_charge_rate(self, t: int) -> float:
        """
        Query the underlying rate_process for the instantaneous
        charging rate (energy per unit time) at time t.
        """
        # print (self._start_offset_ms)
        abs_time = self._start_offset_ms + t
        return self.rate_process.get_rate(abs_time)



    def charge(self, dt: int, t: int) -> float:
        """
        Charge over dt milliseconds, starting at local time t.
        """
        abs_time = self._start_offset_ms + t
        # instantaneous rate at the beginning of the interval
        rate = self.rate_process.get_rate(abs_time)
        energy = rate * dt # (mJ)
        # print(energy)
        added = min(energy, self.capacity - self.current)
        # print(rate)
        self.current += added
        return added


class ArrivalProcess:
    """
    Base interface for arrival processes. On each tick, __call__ returns new tasks.
    Must implement reset() to clear any internal state between episodes.
    """
    def __init__(self):
        pass

    def reset(self) -> None:
        """
        Reset any internal state (e.g. RNG, trace pointer) before a new episode.
        """
        raise NotImplementedError

    def __call__(self, t_old: int) -> List[Task]:
        """
        Generate or retrieve all Task instances arriving at tick t.
        """
        raise NotImplementedError

