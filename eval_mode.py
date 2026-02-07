import pandas as pd
import os
import random
import torch
from cores import cores
from PPO_agent import PPO
from PPO_agent import  SchedulerEnv
from numpy.ma.core import masked
from scheduler import Scheduler
from cores_and_tasks_batt import Battery, Task, SolarChargeRate
from GHI_On_Hottest_Day import irr_profile
from kernel_specs import kernel_specs
from actions import ActionDecoder
from arrivalprocess_taskfactory import PoissonArrivalProcess, build_kernel_param_map, load_arrival_trace
import numpy as np
import matplotlib.pyplot as plt
import os, csv


def _append_metrics_csv(path: str, row: dict, fieldnames: list[str] | None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        # infer a stable order the first time
        fieldnames = list(row.keys())
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
LOG_DIR  = "/home/juiceman/project_apha/Q-scheduler/Q-scheduler_4PART/mainOne/Each_10/eval_log"
LOG_PATH = os.path.join(LOG_DIR, "eval_metrics.csv")

arrival, total_index = load_arrival_trace("arrival_trace.csv",
                                 cores=cores,
                                 kernel_specs=kernel_specs,
                                 use_absolute_deadline=True)

solar = SolarChargeRate(alpha=0.18, area=0.001, irr_profile=irr_profile)
battery = Battery(capacity= 2_000_000.0, rate_process=solar,
                      initial_charge =2000000.0 , begining_time=6*3600*1000)
M_window, H, O_max, L_max, T_window = 20, 900000, 10, 5, 100
feat_len = Task.feature_length(len(cores))
action_decoder = ActionDecoder(M_window, len(cores), D_max=20)
scheduler = Scheduler(cores, battery, arrival,
                      M_window, H, O_max, L_max, T_window, feat_len, total_index)
# print (total_index)
model_path ='Each_10' + '/MaskablePPO'
inputNum_size   = [20, 9, 1, 9]
featureNum_size = [24, 8, 3, 5]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
env = SchedulerEnv(scheduler, action_decoder)
model = PPO(batch_size=681, inputNum_size=inputNum_size,
                  featureNum_size=featureNum_size, device=device)
model.load_using_model_name(model_path)


def RL_OneAction(model,env):
    o, _ = env.reset()

    green_reward=0
    reward = 0
    while True:
        o = o.reshape(1, 20 + 9 + 1 + 9, 24)
        state = torch.FloatTensor(o).to(device)
        mask = env.get_valid_mask()  ## for now just assign_task ! next we need delay as well !
        mask = torch.tensor(np.stack(mask), dtype=torch.bool, device=device)

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)  # (1, 82)

        a=model.eval_action(state,mask)
        o, r, d, truncated, infos = env.step((a,0))
        # if env.scheduler.battery.current <= 0 :
        # print (f'agent actions : {a}, Time : {env.scheduler.t}, '
        #        f'max slck : {env.scheduler.compute_max_idle()},'
        #        f' batt : {env.scheduler.battery.current}')
        reward += r
        # if r != 0:
        #     print( 'time:', env.scheduler.t, "nice_task:", env.scheduler.finish_on_time,
        #           "all execution : ", sum(env.scheduler.runtime_reward_vector), "all energy",
        #           sum(env.scheduler.energy_reward_vector),
        #           "total wipe:", env.scheduler.total_wipe,
        #           "total miss :", env.scheduler.active_queue_miss_counter,
        #           "all task :", env.scheduler.task_name_finished)
        # print (r)
        if d:
            # print('gg')

            break

    return reward
all_metrics = []
fieldnames = None  # infer on first write
def run_policy(iters) :
    fieldnames = None
    marl_r = []
    # seed = 0
    # random.seed(seed)
    for inter_num in range (0, iters) :
        reward = RL_OneAction(model,env)
        marl_r.append(reward)
        metrics = env.scheduler.render(mode='human')

        # augment with eval-specific fields
        metrics = {
            "episode_id": inter_num,
            "eval_return": float(reward),
            **metrics,
        }

        # append to CSV
        if fieldnames is None:
            fieldnames = list(metrics.keys())
        _append_metrics_csv(LOG_PATH, metrics, fieldnames)

        all_metrics.append(metrics)

    return all_metrics



all_metrics_1 = run_policy(100)
# print(all_metrics_1)