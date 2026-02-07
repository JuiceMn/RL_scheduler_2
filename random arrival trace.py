import pandas as pd
import numpy as np

# 1) Your kernels and their exact‐mult runtimes (ms)
kernel_names = [
    "FFT","Susan","BasicMath","Stringsearch","CRC32","AES","Blowfish",
    "Dhrystone","Grayscale","Msort","Norx","Primes","Qsort","SHA256",
    "Sharpen","Square_mmult"
]
rt_exact_list = [
    709308.24, 146710.00, 55010,   530,   23686,  1651.60, 397.57,
      26.63,    89.83, 21.45,   57.09,   2.69,   26.93,   50.50,
      95.57,    95.57
]
rt_map = dict(zip(kernel_names, rt_exact_list))

# 2) Trace parameters
mean_gap         = 17000       # ms defult = 6500
tick_ms          = 1            # ms resolution
horizon_ms       = 3_600_000    # total sim time
max_multiplier   = 5           # allow deadline = k * rt_exact, k∈[1..10] test default was 10
seed             = 42

rng = np.random.default_rng(seed)

# 3) How many tasks?
num_tasks = int(np.ceil(horizon_ms / mean_gap))

# 4) Sample inter‐arrival times & get absolute arrival timestamps
inter_arrivals = rng.exponential(mean_gap, size=num_tasks)
inter_arrivals = np.maximum(1, np.round(inter_arrivals / tick_ms).astype(int))
arrival_times  = np.cumsum(inter_arrivals)
arrival_times  = arrival_times[arrival_times <= horizon_ms]
N = len(arrival_times)

# 5) Pick a kernel for each arrival
task_names = rng.choice(kernel_names, size=N, replace=True)

# 6) For each task, pick a random integer multiplier k
multipliers = rng.integers(2, max_multiplier+1, size=N)

# 7) Compute each task’s deadline = arrival_time + k * rt_exact
rt_for_tasks = np.array([rt_map[name] for name in task_names])
rel_deadlines = (multipliers * rt_for_tasks).astype(int)
deadlines     = arrival_times + rel_deadlines

# 8) Build DataFrame
df = pd.DataFrame({
    "task_name"   : task_names,
    "arrival_time": arrival_times,
    "deadline"    : deadlines, # absolute deadline
    "k_multiplier": multipliers,
    "rt_exact"    : rt_for_tasks
})

# print(df.head(10))


output_path = "/home/juiceman/project_apha/Q-scheduler/Q-scheduler_4PART/mainOne/arrival_trace.csv"
df.to_csv(output_path, index=False)


