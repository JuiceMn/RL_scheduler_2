import bisect
import numpy as np
from typing import List,  Dict, Tuple, Optional, Union
from cores_and_tasks_batt import ArrivalProcess, Task, Core
import pandas as pd
from collections import defaultdict
INF = float('inf')

class PoissonArrivalProcess(ArrivalProcess):
    """
    At each tick t, draws N ~ Poisson(rate) and returns N Tasks:
      – types chosen uniformly from `types`
      – deadlines = t + max(1, round(Normal(mean_deadline, std_deadline)))
      – other fields looked up in `param_map`.
    """
    def __init__(self,
                 rate: float,
                 types: List[str],
                 param_map: dict,
                 mean_deadline: float,
                 std_deadline: float,
                 seed: int = None):
        """
        rate           : λ, mean arrivals per tick
        types          : e.g. ['A','B',…,'P']
        param_map      : dict type→{req_* flags, rt_est, WCEC}
        mean_deadline  : μ for Normal deadline offset
        std_deadline   : σ for Normal deadline offset
        seed           : RNG seed
        """
        super().__init__()
        self.rate = rate
        self.types = types
        self.param_map = param_map
        self.mean_deadline = mean_deadline
        self.std_deadline  = std_deadline
        self.rng = np.random.default_rng(seed)

        # store the “master” seed so we can re-start from it every episode
        self._initial_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset internal RNG.  If `seed` is given, use that;
        otherwise fall back to the original seed you started with.
        """
        use_seed = seed if seed is not None else self._initial_seed
        self.rng = np.random.default_rng(use_seed)

    def __call__(self, t: int, t_new: int) -> List[Task]:
        # 1) how many arrivals this tick?
        num_arrivals = int(self.rng.poisson(self.rate))
        out: List[Task] = []
        for _ in range(num_arrivals):
            # 2) pick a random type
            typ = self.rng.choice(self.types)
            # 3) sample a positive relative deadline
            rel = int(round(self.rng.normal(self.mean_deadline, self.std_deadline)))
            rel = max(1, rel)
            deadline = t + rel
            # 4) lookup parameters for this type
            p = self.param_map[typ]
            task = Task(
                task_name      =typ, ##  key of that dict
                arrival_time   = t,
                deadline       = deadline,
                req_exact_int  = p['req_exact_int'],
                req_approx_int = p['req_approx_int'],
                req_exact_fp   = p['req_exact_fp'],
                req_approx_fp  = p['req_approx_fp'],
                rt_est         = p['rt_est'],
                WCEC           = p['WCEC']
            )
            out.append(task)
        return out

class TraceArrivalProcess(ArrivalProcess):
    """
    Replays a pre‐defined arrival trace: a dict tick -> list of Tasks.
    """
    def __init__(self, trace: Dict[int, List[Task]]):
        super().__init__()
        self.trace = trace
        self._ticks = sorted(trace.keys())
        self.counter = 0
    def delta_to_next_arrival(self, t:float)-> float:
        i = bisect.bisect_right(self._ticks, int(t))
        if i < len(self._ticks):
            return float(self._ticks[i] - t)
        return INF
    def reset(self) -> None:
        # Nothing to reset for a static trace
        pass

    def __call__(self, t_old: int) -> List[Task]:
        # out = []
        # for ti in range(int(t_old+1), int(t_new+1)):
        #     # print (t_old+1, t_new+1)
        #     out.extend(self.trace.get(ti, []))
        # # if out  :
        # #     self.counter +=len(out)
        # #     print(self.counter)
        # #     print (out)
        # return out
        if self.trace.get(t_old, []) != [] :
            # print(self.trace.get(t_old, []))
            self.counter += 1
        if self.counter == 228:
            self.counter = 0
        # print(self.counter)
        return self.trace.get(t_old, [])

def load_arrival_trace(
    df: Union[str, pd.DataFrame],
    *,
    cores: List[Core],
    kernel_specs: List[Dict[str, object]],
    use_absolute_deadline: bool = False
) -> TraceArrivalProcess:
    """
    Reads a CSV (or DataFrame) with columns:
      - arrival_time (int)
      - task_name    (str, one of the kernel_specs names)
      - deadline     (int, absolute tick) **if** use_absolute_deadline=True
        OR
      - rel_deadline (int offset)    **if** use_absolute_deadline=False

    Builds your (types, param_map) via build_kernel_param_map(cores,kernel_specs),
    then for each row creates a Task(arrival_time, deadline, …) and
    buckets it into `trace[arrival_time]`.
    """
    # Load into DataFrame if needed
    if isinstance(df, str):
        df = pd.read_csv(df)

    # 1) Build your types & param_map
    _, param_map = build_kernel_param_map(cores, kernel_specs)

    # 2) Build trace dict
    trace_dict: Dict[int, List[Task]] = defaultdict(list)
    for row in df.to_dict(orient="records"):
        t = int(row["arrival_time"])
        name = row["task_name"]
        if use_absolute_deadline:
            dl = int(row["deadline"])
        else:
            dl = t + max(1, int(row["rel_deadline"]))

        p = param_map[name]
        task = Task(
            task_name      = name,
            arrival_time   = t,
            deadline       = dl,
            req_exact_int  = p["req_exact_int"],
            req_approx_int = p["req_approx_int"],
            req_exact_fp   = p["req_exact_fp"],
            req_approx_fp  = p["req_approx_fp"],
            rt_est         = p["rt_est"],
            WCEC           = p["WCEC"],
        )
        trace_dict[t].append(task)

    # print(sorted(list(trace_dict)))
    # print(len(list(trace_dict)))
    return TraceArrivalProcess(dict(trace_dict)), len(list(trace_dict))


def build_kernel_param_map(
    cores: List[Core],
    kernel_specs: List[Dict[str, object]]
) -> Tuple[List[str], Dict[str, dict]]:
    """
    From your in‐memory `kernel_specs` and `cores` list, build:
      - types:    list of all kernel names
      - param_map: mapping name → { req_..., rt_est, WCEC }
    """
    I = len(cores)
    param_map: Dict[str, dict] = {}

    for spec in kernel_specs:
        name = spec["name"]

        # 1) requirement flags
        req_exact_int  = (spec["int"] == "Exact")
        req_approx_int = (spec["int"] == "Approximate")
        req_exact_fp   = (spec["fp"]  == "Exact")
        req_approx_fp  = (spec["fp"]  == "Approximate")

        # 2) base runtimes
        rt_exact  = float(spec["rt_exact"])
        rt_approx = float(spec["rt_approx"]) if spec["rt_approx"] is not None else None

        # 3) compute per-core rt_est
        rt_est = [np.inf] * I
        for i, core in enumerate(cores):
            # check feasibility
            if req_exact_int  and not core.exact_int:      continue
            if req_exact_fp   and not core.exact_fp:       continue
            if req_approx_int and not (core.approx_int or core.exact_int): continue
            if req_approx_fp  and not (core.approx_fp or core.exact_fp):   continue

            # now decide which runtime that core pays
            if req_exact_int or req_exact_fp:
                # exact‐mult tasks always pay the exact runtime
                rt_est[i] = rt_exact
            elif req_approx_int or req_approx_fp:
                # approximate tasks: if core has approx unit, use rt_approx; else fall back to rt_exact
                if rt_approx is not None and ((req_approx_int and core.approx_int) or
                                              (req_approx_fp and core.approx_fp)):
                    rt_est[i] = rt_approx
                else:
                    rt_est[i] = rt_exact
                if (name == "Susan") and (req_approx_fp and core.approx_fp):
                    rt_est[i] = rt_exact
            else:
                # no‐multiplier tasks can run anywhere at the exact runtime
                rt_est[i] = rt_exact

        # 4) WCEC = runtime × core.avg_power (units: e.g. ms×mW/1000 → mJ)
        WCEC = [(rt_est[i] * cores[i].avg_power/1000) for i in range(I)]


        # 5) store
        param_map[name] = {
            "req_exact_int":  req_exact_int,
            "req_approx_int": req_approx_int,
            "req_exact_fp":   req_exact_fp,
            "req_approx_fp":  req_approx_fp,
            "rt_est":         rt_est,
            "WCEC":           WCEC,
        }

    types = list(param_map.keys())
    return types, param_map
