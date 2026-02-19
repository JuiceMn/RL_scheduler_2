# scheduler.py

import numpy as np
from collections import deque
import math
import itertools

from sympy import false
from torch.distributed.pipelining import pipeline

from cores_and_tasks_batt import Core, Task, Battery
INF = float('inf')
from typing import List, Tuple, Optional, Callable, Deque
from itertools import chain

class Scheduler:
    def __init__(self,
                 cores: list[Core],            # list of Core objects
                 battery: Battery,          # Battery object
                 arrival_process: Callable[[int], List[Task]],  # callable gen for new tasks
                 M,                # window size
                 H,                # look‐ahead horizon
                 O_max,            # overflow‐vector length
                 L_max,            # time‐since‐arrival thermometer length
                 T_window,          # normalization window for t_since_start
                 feature_length ,   # fixed length of each task feature vector
                 total_index
                ):
        # --- Resource state
        self.num_steps = 0
        self.cores = cores
        self.I = len(cores) ## Holds your list of Core objects (each knows its multiplier FUs).
        # self.running_job = [None] * self.I
        # instead of a single slot per core, keep a queue
        self.pipeline: List[Deque[Task]]   = [deque() for _ in range(self.I)]
        self.wait_pipeline: List[Deque[Task]]   = [deque() for _ in range(self.I)]
        self.finished_tasks: List[Task] = [] # finish tracking
        self.deadline_miss_in_active : List[Task] = []
        self.remainingTime:  List[int]      = [0]*self.I ## “How many ticks left until core i is free?” Starts at 0 (idle).
        self.nextFreeTime:   List[int]      = [0]*self.I ## Holds the absolute clock‐time when each core will next be free—used to check look-ahead feasibility beyond just “is it idle right now.”
        self.each_core_util: List[int]      = [0]*self.I
##################################################################################################################################################################################
##################################################################################################################################################################################
        # --- Battery + SSE
        self.hold_reward = 0.1  # constant reward for hold action
        self.charge_penalty = 5.0  # penalty when assigning during charging_flag
        self.battery = battery   # has .capacity, .charge_rate, .current
        self.charging_flag = False
        self.reach_100 = False  # “have we hit full capacity at least once?”
        self.backlog_extend = True  # “may we append new arrivals to backlog?”
        # self.D_max = D_max  ## ?
        # self.bucketWCEC = [0]*(D_max+1) ## Sum of all worst‐case energy (WCEC) for tasks with deadline = d.
        # self.prefixWCEC = [0]*(D_max+1) ##Cumulative sum up to deadline d, used to compute SSE quickly.
##################################################################################################################################################################################
##################################################################################################################################################################################
        # --- Queues
        self.M = M ## Size of your sliding‐window of “visible” waiting jobs.
        self.activeQueue: Deque[Task] = deque(maxlen=M) ## Up to M jobs you feed into the state-vector each tick.
        self.backlog:     Deque[Task] = deque() ## Extra arrived jobs waiting to fill the window when it shrinks.
##################################################################################################################################################################################
##################################################################################################################################################################################
        # --- Arrival process
        self.arrival = arrival_process ## A function you provide that returns a list of new Task objects arriving at time = tick.
##################################################################################################################################################################################
##################################################################################################################################################################################        # --- Look‐ahead + overflow + thermometer
        self.H     = H ## How far ahead (in ticks) you check core availability when validating an assign.
        self.O_max = O_max ## Max length of the “overflow” binary vector.
        self.L_max = L_max ## Max length of the “time‐since‐last‐arrival” thermometer.
        self.t_since_arrival = 0 ## Counts ticks since the most recent new‐job arrival (for that thermometer).
##################################################################################################################################################################################
##################################################################################################################################################################################
        # --- Time tracking
        self.t = 0 ## Global clock (in ticks).
        self.t_since_start = 0 ## “Burst counter” = ticks since last idle period.
        self.T_window = T_window ## ????Used to normalize t_since_start into [0,1]
        self.hold = 0
        self.hold_time = 0
        self.max_deadline = 0
        self.ccmd = ''
        self.valid_vector = [None] * 6
        self.fail_assign = 0

##################################################################################################################################################################################
##################################################################################################################################################################################
        # --- task feathre lenght
        self.feature_length = feature_length
        self.completed_count = 0  # number of tasks finished so far
        self.completed_runtime_sum = 0.0  # sum of actual runtimes of finished tasks
        self.avg_runtime = 0.0  # avg runtime = completed_runtime_sum / completed_count
        self.total_waiting = 0.0
        self.num_assigned = 0
        self.avg_waiting = 0.0
        self.avg_slack = 0.0
        self.total_slack = 0.0
        # for the reward
        self.runtime_reward = 0.0
        self.waiting_reward = 0.0
        self.energy_reward = 0.0
        self.met_deadline_reward = 0.0
        self.runtime_reward_vector = []
        self.waiting_reward_vector = []
        self.energy_reward_vector = []
        self.met_deadline_reward_vector = []
        #####
        self.slack_reward = 0.0
        ########counters#####
        self.active_queue_miss_counter = 0
        self.finish_task_miss_counter = 0
        self.finish_on_time = 0
        self.nowcounter = 0
        self.task_name_finished=[]
        ##### log test
        self.total_execution = 0
        self.total_energy = 0
        self.total_index = total_index
        self.total_wipe = 0
        # self.each_core_util = []
        self.number_two_assign = 0
        self.number_two_eject = 0
        self.number_3_assign = 0
        self.number_3_eject = 0
        self.number_6_assign = 0
        self.number_6_eject = 0
        self.assign_to_2 = []
        self.assign_to_6 = []
        self.negative_slack_count = 0
        self.total_delay_slack = 0
##################################################################################################################################################################################
##################################################################################################################################################################################
    # --- rst

    def reset(self) -> List[float]:
        """Reset everything to start a new episode.
        Resets all clocks, queues, cores, and battery to initial conditions.

        Pulls in any tasks arriving at time 0 into the backlog, then fills the active window via _refill_queue().

        Returns the initial state vector for RL to observe.
        """
        self.t = 0
        self.num_steps = 0
        self.t_since_start = 0
        self.t_since_arrival = 0
        self.num_assigned = 0
        self.runtime_reward = 0
        self.waiting_reward = 0
        self.energy_reward = 0.0
        self.met_deadline_reward = 0.0
        self.runtime_reward_vector = []
        self.waiting_reward_vector = []
        self.energy_reward_vector = []
        self.met_deadline_reward_vector = []
        self.slack_reward = 0
        self.avg_waiting = 0
        self.avg_slack = 0
        self.avg_runtime= 0
        self.total_waiting = 0
        self.total_slack = 0
        self.finish_task_miss_counter = 0
        self.finish_on_time = 0
        self.total_wipe = 0
        self.active_queue_miss_counter = 0
        for i in range(self.I):
            self.pipeline[i].clear()
            self.wait_pipeline[i].clear()
            self.remainingTime[i] = 0
            self.nextFreeTime[i]  = 0
            self.each_core_util[i] = 0
        # self.battery.current = self.battery.capacity
        if self.battery.initial is  not None :
            self.battery.current =  self.battery.initial
        else :
            self.battery.current = float(0)
        # print(len(self.activeQueue), len(self.backlog))
        self.activeQueue.clear()
        self.backlog.clear()
        self.finished_tasks.clear()
        self.task_name_finished = []
        self.total_execution = 0
        self.total_energy = 0
        self.negative_slack_count = 0
        self.total_delay_slack = 0
        self.number_two_assign = 0
        self.number_two_eject = 0
        self.number_6_assign = 0
        self.number_6_eject = 0
        self.fail_assign = 0
        self.assign_to_2 = []
        self.assign_to_6 = []
        # pre‐fill arrivals at t=0
        for job in self.arrival(self.t):
            self.backlog.append(job)
        self._refill_queue()
        return self.build_state()
    ##################################################################################################################################################################################
    ##################################################################################################################################################################################
    def build_state(self) -> List[float]:
        """Return the full state vector as a flat list or array.
        Part A: For each core i, append its 4-bit capability vector + 1 scalar remainingTime
        """
        state = []
        # D) Queue window: up to M tasks

        ##Part D: For each of the M slots
        ## f a job is present, call job.feature_vector(t) (should return 24 dims).
        ## esle pad with zeros.
        for idx in range(self.M):

            if idx < len(self.activeQueue):
                # print('---_', idx)
                job = self.activeQueue[idx]
                vec = job.feature_vector(self.t)
                waiting1 = self._norm_time(vec[0])
                # ignore old slack vec[1]
                bits = vec[2:6]  # four bits
                rt_est_vals = [self._norm_time(x) for x in vec[6:6 + self.I]]  # I floats
                WCEC_vals = [self._norm_energy(x) for x in vec[6 + self.I:6 + 2 * self.I]]  # I floats
                ## using simple job slck for avoididng bugs !!!
                new_slack = max (0, self._norm_time(job.slack_time))
                state.extend([
                    waiting1,
                    new_slack,
                    *bits,
                    *rt_est_vals,
                    *WCEC_vals,
                ])
                # print([
                #     waiting1,
                #     new_slack,
                #     *bits,
                #     *rt_est_vals,
                #     *WCEC_vals,
                # ])
            else:
                state.extend([0.0] * self.feature_length)

        # Running_tasks :
        for idx in range(self.I):
            if self.pipeline[idx]:
                job, exec_t, finish_t, _ = self.pipeline[idx][0]
                remaining_time = self._norm_time(finish_t - self.t)
                vec = job.feature_vector(self.t)
                waiting = self._norm_time(vec[0])
                # ignore old slack vec[1]
                ## now I sue mni-raw compute_job_slack(job) maybe I should use remaininig_time insteasd !
                bits = vec[2:6]
                ## using simple job slck for avoididng bugs !!!
                state.extend([waiting, max(0, self._norm_time(job.slack_time)), *bits, remaining_time,
                              self._norm_energy(job.WCEC[idx])])
                # state.extend([0.0] *(self.feature_length-5))
                # print([waiting, self.compute_job_slack(job), *bits, remaining_time, job.WCEC[idx] ], '---------> RUNNING')
            else:
                state.extend([0.0] * 8)
            state.extend([0.0] * 16)
        # B) Battery + SSE
        """
        Part B:

        Normalized State‐Of‐Charge
        Normalized charging rate
        Normalized SSE (computed in _compute_SSE
        """
        # charging_flag = self.charging_flag
        soc_norm = self.battery.current / self.battery.capacity
        r_norm = self.battery.get_charge_rate(self.t) / self.battery.capacity
        sse, _ = self._compute_SSE()
        sse_norm = sse / self.battery.capacity
        # state += [charging_flag,
        # state += [soc_norm, r_norm, sse_norm]
        # print (soc_norm, sse)
        state += [
            float(max(0.0, min(1.0, soc_norm))),
            float(max(0.0, min(1.0, r_norm))),
            float(max(0.0, min(1.0, sse_norm))),
        ]
        state.extend([0.0] * 21)



        # A) Cores: caps + remainingTime
        for i, core in enumerate(self.cores):
            state.extend(core.capabilities_bits())   # 4 bits
            state.append(self._norm_time(self.remainingTime[i]))      # 1 scalar
            state.extend([0.0] * 19)
##################################################################################################################################################################################
##################################################################################################################################################################################

##################################################################################################################################################################################
##################################################################################################################################################################################
        # # C) Optional: t_since_start normalized
        # """
        #    Part C: One scalar = “normalized burst counter
        # """
        # t_norm = min(self.t_since_start, self.T_window) / self.T_window
        # state.append(t_norm)
        # fellan zaman hazf mishe 
##################################################################################################################################################################################
##################################################################################################################################################################################

##################################################################################################################################################################################
##################################################################################################################################################################################

##################################################################################################################################################################################
##################################################################################################################################################################################
 ### comented for test pupose
#         # E) Overflow vector
#         """
#         Omax-lenght length thermometer of 1’s = how many tasks are beyond your window.
#         """
#         overflow = max(0, len(self.activeQueue)+len(self.backlog)-self.M)
#         ones = min(overflow, self.O_max)
#         state += [1]*ones + [0]*(self.O_max-ones)
# ##################################################################################################################################################################################
# ##################################################################################################################################################################################
#         # F) Time‐since‐arrival thermometer
#         """
#         lmax-length thermometer counting ticks since last arrival.
#         """
#         ones = min(self.t_since_arrival, self.L_max)
#         state += [1]*ones + [0]*(self.L_max-ones)
##################################################################################################################################################################################
##################################################################################################################################################################################
        return state

##################################################################################################################################################################################
##################################################################################################################################################################################

    #################compute_SSE############
    def _compute_SSE(self) -> Tuple[float, float]:
        # If there are literally no pending jobs anywhere, just return current SoC
        if not self.activeQueue and not self.backlog:
            # print('exit1')
            return self.battery.current, 0 # just for test


        # 1) Pool both active‐window tasks and backlog tasks
        t1= self.t
        all_tasks = list(chain(self.activeQueue, self.backlog))
        # print(all_tasks)

        # 2) Compute the furthest deadline among them
        max_deadline = max(job.deadline for job in all_tasks)
        # print(max_deadline)

        # 3) Keep only those whose deadline falls within your look‐ahead horizon
        jobs = [j for j in all_tasks]
                #if j.arrival_time >= t1 and j.deadline <= max_deadline ]

        # print(jobs)
        if not jobs:
            # print('exit2')
            return self.battery.current, 0
                                    # just for test

############################calssic method##########################################
        # # 4) Stack requirement‐bits and WCEC
        # Req = np.array([
        #     [j.req_exact_int, j.req_approx_int, j.req_exact_fp, j.req_approx_fp]
        #     for j in jobs
        # ], dtype=bool)  # shape (N,4)
        # W = np.array([j.WCEC for j in jobs], dtype=float)  # shape (N,I)
        # # 5) Core capability matrix
        # Caps = np.array([
        #     core.capabilities_bits()
        #     for core in self.cores
        # ], dtype=bool)  # shape (I,4)
        # feas = ~(Req[:, None, :] & ~Caps[None, :, :]).any(axis=2)  # shape (N,I)
        # # 7) For each job pick its cheapest feasible WCEC
        # # min_w = np.min(np.where(feas, W, np.inf), axis=1)  # shape (N,)
        # feasible_W = np.where(feas, W, np.nan)  # shape (N,I)
        # avg_w = np.nanmean(feasible_W, axis=1)  # shape (N,)
        # g = float(np.nansum(avg_w))
        # g = float(min_w.sum())
##########################new_one############################################
            # 6) Build your new feasibility mask
        feas = np.array([
            [self.check_capability(job, core)
             for core in self.cores]
            for job in jobs
        ], dtype=bool)  # shape (N,I)
        # 7) Compute per‐job average WCEC over *those* feasible cores
        W = np.array([np.round(j.WCEC, 3) for j in jobs], dtype=float)  # (N,I)
        finite_w = W.copy()
        finite_w[~feas] = 0
        num_feas = feas.sum(axis=1)  # (N,)
        # avg_wcec = finite_w.sum(axis=1) / num_feas
        max_wcec = finite_w.max(axis=1)
        # print(max_wcec)
        # print('**************')
        # 8) Sum up reserved energy
        g = float(max_wcec.sum())
        # print(g)
        # print("==^" * 50)
###############################################################


        # 9) Predict total incoming energy until the furthest deadline
        t1_ms = self.t
        t2_ms = max(job.deadline for job in all_tasks)
        # print(t2_ms)
        delta_ms = t2_ms - t1_ms
        # print(delta_ms)
        # how many 15-min slots fit in that interval?
        # (round up so partial slots still count)
        interval = self.battery.rate_process.interval_ms
        horizon_slots = int((delta_ms + interval - 1) // interval)
        # print(horizon_slots)
        # call your solar‐forecast once
        abs_t1 = self.battery._start_offset_ms + t1
        incoming_rates = self.battery.rate_process.forecast(abs_t1, horizon_slots)
        # print(incoming_rates)
        incoming = sum(incoming_rates) * delta_ms
        # print(f'incoming : {incoming}, deta_t:{delta_ms}, that job deadline : {t2_ms}')
        # print(incoming)
        # print('*' * 40)

        # 10) SSE = current + incoming − reserved
        SSE = min (self.battery.capacity, self.battery.current + incoming) - g
        # print(f'SSE:{SSE}, ---, battery.current:{self.battery.current}, ---, incoming:{incoming}, ---, g:{g}')
        # print(self.battery.current)
        # print(SSE)
        # T_energy_slack = SSE / incoming if incoming > 0 else 100000 # just for test
        # print(SSE)
        if SSE >= 0 :
            T_energy_slack = 0
        else :
            self.negative_slack_count += 1
            t2_temp = t2_ms
            SSE_temp = SSE
            # print (SSE_temp)
            while SSE_temp < 0 :
                t2_temp+=1
                delta_temp = t2_temp - t1_ms
                horizon_slots_temp = int((delta_temp + interval - 1) // interval)
                incoming_rates_temp = self.battery.rate_process.forecast(abs_t1, horizon_slots_temp)
                incoming_temp = sum(incoming_rates_temp) * delta_temp
                SSE_temp = min (self.battery.capacity, self.battery.current + incoming_temp) - g
            T_energy_slack = delta_temp
        # print (f'sse : {SSE}')
        # print('exit3')
        # print(T_energy_slack)
        return SSE, T_energy_slack

    ##################################################################################################################################################################################
##################################################################################################################################################################################
    def check_assign(self, i: int, m: int) -> Optional[bool]:
        """
        Returns None if assign(i,m) is valid, else a string reason code:
         - 'overflow'        window overflow
         - 'capability'      multiplier req vs. core cap
         - 'timing'          lookahead deadline
         - 'runtime'         INF (impossible runtime)
         - 'energy'          instant battery
         - 'budget'          SSE/future-budget
        """

        if m >= len(self.activeQueue):
            # print('out of range !')
            # self.valid_vector[0] = 1
            # print('m >= len(self.activeQueue)')
            return False
        job  = self.activeQueue[m]
        core = self.cores[i]

        if not self.check_capability (job, core):
            # print('bitwise')
            # self.valid_vector[1] = 1
            return False

        # 1) multiplier‐requirement vs. capability
        # req = np.array([job.req_exact_int, job.req_approx_int,
        #                 job.req_exact_fp,  job.req_approx_fp], dtype=bool)
        # cap = np.array(core.capabilities_bits(), dtype=bool)
        # # print(req, job.task_name)
        # # print(cap, i)
        # # print (np.any(req & ~cap))
        # # print('='*30)
        # if np.any(req & ~cap):
        #     print('bitwise')
        #     return False

        # 2) timing look-ahead + impossible runtime
        start  = max(self.t, self.nextFreeTime[i])
        finish = start + job.rt_est[i]
        if job.rt_est[i] == INF:
            # self.valid_vector[2] = 1
            return False




        # omited for test !
        # if finish > self.t + self.H :
        #     # print ('damn H')
        #     # self.valid_vector[3] = 1
        #     return False

        # 3) instant battery
        # print(job.task_name, job.WCEC[i], self.battery.current)
        if self.battery.current < job.WCEC[i]:

            # print ('damn charge')
            # self.valid_vector[4] = 1
            # print('self.battery.current < job.WCEC[i]')
                # print ("naughty reach here ! ")
            return False


        # if job.deadline >= job.rt_est[i] :
        #     # maybe there is at least one better core to run !
        #     return False
        # # 4) SSE / future-budget
        # if self._compute_SSE() - job.WCEC[i] < 0:
        #     return 'budget'
        # print('we reach this shit !?')
        # print(m, 'belong to before valid assign ')
        #######################################################################################
        #######################################################################################
        if job.slack_time - job.rt_est[i] > 0 : # don't waste my time
            if self.pipeline[i] :
                if job.deadline > self.pipeline[i][0][0].deadline :
                    # print(job.deadline, self.pipeline[i][0][0].deadline)
                    execution_time = job.rt_est[i]  # execution time on core i
                    finish_time = max(self.t, self.nextFreeTime[i]) + execution_time  # actual finish time
                    if job.deadline < finish_time : # maybe there is a better core to run !
                        # print('cant run')
                        return False
                else :
                    finish_time = self.t + job.rt_est[i]
                    if job.deadline < finish_time :
                        # print('cant run_1')
                        return False
                    #####################################
            else :
                execution_time = job.rt_est[i]  # execution time on core i
                finish_time = max(self.t, self.nextFreeTime[i]) + execution_time  # actual finish time
                if job.deadline < finish_time :
                    # print('cant run_2')
                    return False
        else :
            self.total_wipe+=1
            del self.activeQueue[m]
            self._refill_queue()
            return False


        return True  # all good


    ##################################################################################################################################################################################
##################################################################################################################################################################################
    def assign(self, i: int, m: int, delay: int) -> bool:


        if self.activeQueue :
            job = self.activeQueue[m]
        #
        # if job.rt_est[i] == 23686 or job.rt_est[i] == 530:
        #     # print(job.task_name, 'fff')
        #     i = 0
        # print(job.task_name, i + 1)
        if job.slack_time - job.rt_est[i]> 0 : # don't waste my time
            if self.pipeline[i] :
                if job.deadline > self.pipeline[i][0][0].deadline :
                    # print(job.deadline, self.pipeline[i][0][0].deadline)
                    energy_consumption = job.WCEC[i]
                    execution_time = job.rt_est[i] + delay  # execution time on core i
                    finish_time = max(self.t, self.nextFreeTime[i]) + execution_time  # actual finish time
                    if job.deadline >= finish_time : # maybe there is a better core to run !
                        self.pipeline[i].append([job, execution_time, finish_time, energy_consumption])  # Appending the tuple
                        ####################################
                        start_time = max(self.t, self.nextFreeTime[i])
                        end_time = start_time + job.rt_est[i]
                        self.nextFreeTime[i] = end_time
                        self.remainingTime[i] = end_time - self.t
                        #####################################
                        # job.start_time = start_time
                        job.finish_time = end_time
                        ####################################
                        # self.battery.current = max(0.0, self.battery.current - job.WCEC[i])
                        del self.activeQueue[m]
                        self._refill_queue()
                        return True
                    else :
                        self.fail_assign += 1
                        # print("we reach this shit whole_1")
                        self._advance_time()

                else :
                    finish_time = self.t + job.rt_est[i]
                    if job.deadline >= finish_time :
                        for item in self.pipeline[i]:
                            item[0].finish_time += job.rt_est[i]
                            item[2] += job.rt_est[i]
                        energy_consumption = job.WCEC[i]
                        execution_time = job.rt_est[i] + delay # execution time on core i
                        self.pipeline[i].appendleft([job, execution_time, finish_time, energy_consumption])
                        start_time = self.t
                        end_time = start_time + job.rt_est[i]
                        # this 2 need to check
                        self.nextFreeTime[i] += job.rt_est[i]
                        self.remainingTime[i] += job.rt_est[i]
                        job.finish_time = end_time
                        del self.activeQueue[m]
                        self._refill_queue()
                        return True
                    #####################################
                    # job.start_time = start_time

                    else :
                        self.fail_assign += 1
                        # print("we reach this shit whole_2")
                        self._advance_time()
            else :
                energy_consumption = job.WCEC[i]
                execution_time = job.rt_est[i] + delay # execution time on core i
                finish_time = max(self.t, self.nextFreeTime[i]) + execution_time  # actual finish time
                if job.deadline >= finish_time :
                    self.pipeline[i].append([job, execution_time, finish_time, energy_consumption])  # Appending the tuple
                    ####################################
                    start_time = max(self.t, self.nextFreeTime[i])
                    end_time = start_time + job.rt_est[i]
                    self.nextFreeTime[i] = end_time
                    self.remainingTime[i] = end_time - self.t
                    #####################################
                    # job.start_time = start_time
                    job.finish_time = end_time
                    ####################################
                    # self.battery.current = max(0.0, self.battery.current - job.WCEC[i])
                    del self.activeQueue[m]

                    return True
                else :
                    self.fail_assign += 1
                    # print("we reach this shit whole_3")
                    self._advance_time()
        else :
            # print("we reach this shit whole_4")
            self.fail_assign += 1
            self._advance_time()
            # self.total_wipe+=1
            # del self.activeQueue[m]
        # print ("can we reach here?")

            return False



##################################################################################################################################################################################
##################################################################################################################################################################################
    def step(self, action1: Tuple[str, int, int], action2: int) :
        # print(action1)
        cmd, i, m = action1
        self.ccmd = cmd
        delay = action2
        self.num_steps += 1

        # Case 1: a valid assignment
        # print(m, '-------------->', 'belong to before giving to valid for last check assign ')


        if cmd == 'hold':
            # # print(cmd, delay)
            # # mark that we held this timestep
            # self.hold = 1 if self.compute_max_idle() != 0 else 0 #  now we are good ! =>stuck at zero (gnd)
            # # advance time once
            # if self.compute_max_idle() != 0 :
            #     self.total_delay_slack += self.compute_max_idle()
            # self.hold_time = self.compute_max_idle()
            self._advance_time()
            # use your hold reward
            # reward = self.hold_reward

        elif cmd == 'assign' and self.check_assign(i, m) :
            # perform assignment exactly as before
            # print(m,i, len(self.activeQueue), len(self.backlog), cmd, self.t)
            check_fail_assign = False
            job_delay = self.activeQueue[m]
            # if job_delay.naughty_task :
            #     job_delay.naughty_task = False
            #     job_delay.rt_est[i] -= job_delay.task_delay
            #     job_delay.task_delay = 0

            if delay != 0 :
                # job_delay.naughty_task = True
                job_delay.task_delay = delay
                # job_delay.rt_est[i]+= delay
            check_fail_assign = self.assign(i, m, delay)
            if check_fail_assign :
                self.fail_assign = 0
            # self._advance_time() tested
            # base_reward = self._compute_reward()
            # penalty = self.charge_penalty if self.charging_flag else 0.0
            # reward = base_reward - penalty

        # Case 2: agent chose hold
        # Case 3: invalid assign – treat like hold but without setting hold flag twice
        else:
            # print('ff')
            self._advance_time()
            # reward = self.hold_reward

        state  = self.build_state()
        done   = self._is_done()
        reward, _, _, _ = self._compute_reward()
        info = {
            'episode': {
                'r': reward,
                'length': self.num_steps,
                'time': self.t,
                'state' : self.ccmd
                # 'valid' : self.valid_vector
            },
            'bad_transition': False
        }
        # print (cmd)
        # if i is not None and m is not None :
        #     print(i*m)
        garbage = 0
        # print(self.valid_vector)
        self.valid_vector.clear()
        # print("==========================")
        ### reward just for last last episode just for ppo !
        return state, (reward if done else 0), done, garbage,  info

    ##################################################################################################################################################################################
##################################################################################################################################################################################
    # def _advance_time(self):
    #     self.t += 1
    #     for i in range(self.I):
    #         if self.remainingTime[i] > 0:
    #             self.remainingTime[i] -= 1
    #             # if the very next job in pipeline just hit its finish_time
    #             if (self.pipeline[i]
    #                     and getattr(self.pipeline[i][0], 'finish_time', None) == self.t):
    #                 job = self.pipeline[i].popleft()
    #                 self._finish_core(job)
    #     delta = self.battery.get_charge_rate(self.t)
    #     self.battery.current = min(self.battery.capacity,
    #                                self.battery.current + delta)
    #     new_jobs = self.arrival(self.t)
    #     if new_jobs:
    #         self.backlog.extend(new_jobs)
    #         # sort backlog by increasing deadline
    #         self.backlog = deque(sorted(self.backlog, key=lambda job: job.deadline))
    #         self.t_since_arrival = 0
    #     else:
    #         self.t_since_arrival += 1
    #     if any(rt > 0 for rt in self.remainingTime) or self.activeQueue:
    #         self.t_since_start += 1
    #     else:
    #         self.t_since_start = 0
    #     self._refill_queue()
    #     return self.build_state()
##################################################################################################################################################################################

    def get_remaining_times(self):
        """
        This function calculates the remaining time for tasks in all cores' pipelines.
        It returns a list of remaining times.
        """
        remain = []  # List to store the remaining time for tasks in all cores

        remain_delay_last = []

        for i in range(self.I):
            if self.pipeline[i]:
                # Get the finish time of the first task in the pipeline
                _, _, finish_t, _ = self.pipeline[i][0]
                remaining_time = finish_t - self.t
                # Add the remaining time to the remain list
                remain.append(remaining_time)
            remain_delay = []


        return remain

    ##################################################################################################################################################################################
    def _advance_time(self):
        # print(self.t)
        """
        Advance time either by a hold interval (if self.hold == 1) or by the next
        task‐completion interval.  In both cases, pop finished jobs, charge the battery,
        process new arrivals, update counters, and refill the active window.
        """
        # 1) Determine the elapsed interval dt
        # print(self._compute_SSE())
        # print(f'time : {self.t}, battey : {self.battery.current}')
        #       f'sse : {self._compute_SSE()}'
        #       f'backlog : {len(self.backlog)}, active queue : {len(self.activeQueue)}'
        #       f'cores remaining : {self.remainingTime}'
        #       )
        if self.hold == 1:
            # Use the hold_time (must be set elsewhere) as dt
            dt = self.hold_time
            # reset the hold flag
            self.hold = 0

        else:
            # Normal mode: jump to next completion
            remain = self.get_remaining_times()
            dt_temp = min(remain) if remain else INF
            if hasattr(self.arrival, "delta_to_next_arrival"):
                next_arrival_dt = self.arrival.delta_to_next_arrival(self.t)
            else:
                next_arrival_dt = INF  # fallback if using a plain callable
            # print(f'next_arrival_dt: {next_arrival_dt}, dt_temp: {dt_temp}')
            dt = min(dt_temp, next_arrival_dt)
            # dt = min (dt_temp, next_arrival_dt, self.compute_max_idle())
            # print(dt_temp, next_arrival_dt, self.compute_max_idle())
            if not math.isfinite(dt) or dt <= 0:
                dt = 1  # idle tick fallback to avoid stalling
        # *) updating Horizon
        self.H += dt ## the logic should be tested
        # 2) Charge the battery over dt
        self.battery.charge(dt, self.t)
        # print (self.battery.current)
        # print (self.battery.current)
        # print(self.battery.charge(dt, self.t), self.battery.current)

        # 3) Advance the clock
        old_t = self.t
        self.t += dt
        # print(self.t, self.compute_max_idle() ,self.battery.current)
        # 4) Pop any finished jobs from each core’s pipeline
        # for loop and removing from the list buge warning !
        for i in range(self.I):
            if self.remainingTime[i] > 0:
                # print(self.remainingTime, self.t)
                self.remainingTime[i] = max (0, (self.remainingTime[i] - dt))
            if self.pipeline[i]:
                job, exec_t, finish_t, energy_consumption = self.pipeline[i][0] # energy_consumption was granted as well
                remaining = finish_t - self.t
                if remaining <= 0:
                    self.each_core_util[i] += exec_t
                    self.battery.current = max(0.0, self.battery.current - energy_consumption)
                    # print (self.battery.current,'xx', energy_consumption,'xx',job.task_name, i )
                    self.pipeline[i].popleft()
                    self._finish_core(job, energy_consumption, exec_t)
            # if self.wait_pipeline[i] :
            #     for task in self.wait_pipeline[i]:
            #         if task.task_delay > 0:
            #             task.task_delay = max(0, task.task_delay - dt)


        # print(self.remainingTime)
        # 5) New arrivals if allowed
        new_jobs = self.arrival(self.t)
        if self.backlog_extend and new_jobs:
            self.backlog.extend(new_jobs)
            self.backlog = deque(sorted(self.backlog, key=lambda job: job.deadline)) # need to be tested !
            self.max_deadline = self.backlog[-1].deadline if self.backlog else None
            self.t_since_arrival = 0
        else:
            self.t_since_arrival += 1

        # deadline miss ! omited for test !
        # for idx in range(self.M):
        #     if idx < len(self.activeQueue):
        #         job_b = self.activeQueue[idx]
        #         # print(self.compute_job_slack(jobb))
        #         if job_b.slack_time < 0 :
        #             # print(jobb.deadline, jobb.arrival_time, jobb.task_name)
        #             self.total_wipe+=1
        #             self.deadline_miss_in_active.append(job_b)
        #             del self.activeQueue[idx]

        # 6) Update burst counter
        if any(rt > 0 for rt in self.remainingTime) or self.activeQueue:
            self.t_since_start += 1
        else:
            self.t_since_start = 0

        # 7) Refill active window
        self._refill_queue()
        # print(len(self.activeQueue), len(self.backlog))


    #######################################################
    def _check_battery_and_sse(self):
        """
        1) Compute SSE; if it’s negative, stop extending backlog this tick.
        2) If battery <= 20 and we’ve never hit 100% before, start charging mode.
        """
        sse, _ = self._compute_SSE()
        # 1) block or allow backlog growth
        if not self.charging_flag :
            self.backlog_extend = (sse <= 0)

        # 2) track charging_flag & reach_100
        if self.battery.current <= 10.0 and not self.reach_100:
            self.charging_flag = True
    #######################################################
    def _refill_queue(self):
        # print(len(self.backlog))
        while len(self.activeQueue) < self.M and self.backlog:
            job = self.backlog.popleft()
            # print(self.t - job.arrival_time)
            self.activeQueue.append(job)
            # print ("len is : ")
            # print(len(self.activeQueue))
            # self.nowcounter += 1
            # print(self.nowcounter, self.t, job.task_name)
##################################################################################################################################################################################
##################################################################################################################################################################################

    def _finish_core(self, job, ent, exc):
        if job is None:
            return  # nothing was actually running
        self.finished_tasks.append(job)
        # if job.slack_time < 0 :
        #     self.active_queue_miss_counter+=1
        #     print("name :", job.task_name, "slack :", job.slack_time, "total miss:", self.active_queue_miss_counter)
        #######################
        job.finished_on_time = job.finish_time <= job.deadline
        # print(job.deadline)
        if job.finished_on_time:
            # print(job.slack_time)
            self.task_name_finished.append(job.task_name)
            self.finish_on_time += 1
        ### computing_reward :
        # print (self.finish_on_time)
        Energy = ent
        execution = exc
        # print(job.task_name, ':', exc)
        vec = job.feature_vector(self.t)
        waiting = abs(vec[0] - execution)
        # print(waiting)
        ##
        self.runtime_reward_vector.append(execution)
        self.waiting_reward_vector.append(waiting)
        self.energy_reward_vector.append(Energy)
        self.met_deadline_reward_vector.append(1 if (job.finish_time <= job.deadline) else 0)

        self.total_execution += execution
        self.total_energy += ent
        self.num_assigned += 1
        self.waiting_reward = sum(self.waiting_reward_vector)/len(self.waiting_reward_vector)
        self.runtime_reward=sum(self.runtime_reward_vector)/len(self.runtime_reward_vector)
        self.energy_reward=sum(self.energy_reward_vector)/len(self.energy_reward_vector)
        self.met_deadline_reward = sum(self.met_deadline_reward_vector)#/len(self.met_deadline_reward_vector)
        # print (f'total shit here : {self.waiting_reward + self.energy_reward + self.runtime_reward}')
        # Energy Instead of time
        # self.runtime_reward = self.avg_runtime + max(10, Energy)   # threshold for reward
        ## now I sue mni-raw compute_job_slack(job)  !
        # print (self.compute_job_slack(job))
        # back to the root using simple job slack for avoiding bugs !
        # self.slack_reward = self.compute_job_slack(job) + self.avg_slack
        # self.slack_reward = job.slack_time + self.avg_slack
        # ## update_total/data
        # self.total_waiting += waiting
        # delta = Energy - self.avg_runtime
        # self.total_slack += job.slack_time
        # ## update_avg
        # self.avg_runtime += delta / self.num_assigned
        # self.avg_waiting = self.total_waiting / self.num_assigned  # update the avg
        # self.avg_slack = self.total_slack / self.num_assigned

##################################################################################################################################################################################
#########################################################################################

    def _compute_reward(self) -> [float, float, float, float]:
        # blending weights
        alpha, beta, gamma = 0.4, 0.3, 0.3
        gg = 1000
        c         = float(self.met_deadline_reward)
        # print(c)
        runtime_t = float(self.runtime_reward)
        waiting_t = float(self.waiting_reward)
        energy_t   = float(self.energy_reward)
        total = c #- (alpha*runtime_t + gamma*waiting_t + beta*energy_t)/ gg

        return  total, runtime_t, waiting_t, energy_t
#################slack_per_job#############################################################################################################
    def compute_job_slack(self, job: Task) -> float:
        """Compute slack = time‐until‐dl minus earliest‐possible finish time."""
        t0 = self.t
        # build requirement mask

        feas = np.array(
            [self.check_capability(job, core)
             for core in self.cores], dtype=bool)
        # print(feas, job.task_name)
        # if no core can run it, slack = -inf (or large negative)
        # need to check why it can't run on any core 1

        if not feas.any():
            print('no core to run ! ')
            return 0
        # compute finish times on each feasible core
        demands = []
        for i, ok in enumerate(feas):
            if not ok:
                continue
            wait_i   = self.remainingTime[i]
            demand_i = wait_i + job.rt_est[i]
            demands.append(demand_i)
            # print(demands, '---', i)
        h =np.min(np.array(demands))
        # print (h, '****')
        # if sum (self.remainingTime ) != 0 :
        #     print(self.remainingTime, '%%%%%%')


        # print(job.deadline, t0, h)
        x = (job.deadline - t0) - h
        # print(f'time_slack:{x}, dead line = {job.deadline}, now : {t0}, demand : {h}, name : {job.task_name}')
        return x


##################################################################################################################################################################################
#########################################################################################

    def compute_max_idle(self) -> int:
        """
        Δ = min(
           ⌊SSE / instant_charge_rate⌋,
           min_j job_slack(j)
        )
        where j ranges over both activeQueue and backlog.
        """
        # --- 1) energy‐based idle horizon ---
        _, max_idle_energy = self._compute_SSE()



        # --- 2) job‐slack‐based horizon ---
        all_jobs = itertools.chain(self.activeQueue, self.backlog)
        #### using simple job slack for avoiding bugs !!!!
        slacks = [j.slack_time for j in all_jobs]
        if slacks:
            max_idle_slack = (min(slacks))
        else:
            # no jobs at all ⇒ no deadline constraint
            # max_idle_slack = 1000000
            return 0
        sse_temp , _ = self._compute_SSE()
        # print (f"sse is : {sse_temp}")
        if sse_temp < 0 :
            # --- 3) take the more conservative bound, but non‐negative ---
            # print(max_idle_energy)
            # print(f'max_idle_slack = {max_idle_slack}, max_idle_energy = {max_idle_energy}, ')
            # print (f'slack energy : {max_idle_energy}')
            if max_idle_slack < 0 :
                slack_time = max_idle_energy
            else :
                slack_time = max(0, min(math.floor(max_idle_energy),max (0, max_idle_slack)))
            delta = slack_time
            # print(max_idle_energy)
            return delta
        else :
            return 0
########################Done#####################
    def _is_done(self) -> bool:
        # e.g. fixed horizon
         #default was 3600000
        # return  len(self.activeQueue)==0  and len(self.backlog)==0 and self.t>= 4000000
        # print (self.total_index , self.total_wipe, self.num_assigned)
        if self.fail_assign >= 10000 :
            return True
        return  self.num_assigned >= self.total_index - self.total_wipe # default
    def check_capability (self, job, core)-> Optional[bool]:
        # build boolean arrays of length 4
        req = np.array([
            job.req_exact_int,
            job.req_approx_int,
            job.req_exact_fp,
            job.req_approx_fp,
        ], dtype=bool)
        cap = np.array(core.capabilities_bits(), dtype=bool)

        # 1) if there are no requirements at all, it’s always fine
        if not req.any():
            return True

        # 2) compute “ok” masks for each category
        int_ok = cap[0] | cap[1]  # core has either exact‐int or approx‐int
        fp_ok = cap[2] | cap[3]  # core has either exact‐fp  or approx‐fp

        # 3) build four failure conditions
        exact_int_fail = req[0] & ~cap[0]  # needed exact‐int but core lacks it
        exact_fp_fail = req[2] & ~cap[2]  # needed exact‐fp  but core lacks it
        approx_int_fail = req[1] & ~int_ok  # needed approx‐int  but core lacks any int
        approx_fp_fail = req[3] & ~fp_ok  # needed approx‐fp  but core lacks any fp

        # 4) if *any* of those is true, capability check fails
        if exact_int_fail or exact_fp_fail or approx_int_fail or approx_fp_fail:
            return False

        return True

    def render(self, mode='human'):
        """
        Report high-level episode metrics.
        If mode='human' -> pretty print.
        Otherwise returns a dict you can log to file/JSON.
        """
        makespan = float(self.t)  # total simulated time
        makespan2 = float(self.t)  # total simulated time
        completed = int(self.num_assigned)
        finished_on_time = int(self.finish_on_time)
        missed = completed - finished_on_time
        dropped = int(self.total_wipe)

        # Protect against divide-by-zero
        eps = 1e-9
        throughput = completed / max(makespan, eps)
        util = self.total_execution / max( self.I *makespan, eps)
        utilpercore = np.array(self.each_core_util, dtype=float) / (max(makespan2, eps))
        miss_rate_completed = missed / max(completed, 1)
        miss_rate_including_drops = (missed + dropped) / max(self.total_index, 1)
        slack_counter = self.negative_slack_count
        total_delay_slack = self.total_delay_slack

        metrics = {
            "time_now": makespan,
            "makespan": makespan,  # identical at episode end
            "num_cores": self.I,
            "completed": completed,
            "finished_on_time": finished_on_time,
            "missed_deadline": missed,
            "dropped_or_wiped": dropped,
            "slack_counter": slack_counter,
            "slack_delay" : total_delay_slack,
            "deadline_miss_rate_completed": miss_rate_completed,
            "deadline_miss_rate_including_drops": miss_rate_including_drops,
            "throughput_tasks_per_time": throughput,
            "total_energy_consumption": float(self.total_energy),
            "total_core_busy_time": float(self.total_execution),
            "system_utilization": float(util),  # 0..1
            "cores_utilization_json": utilpercore.tolist(),
            # Optional extras you already compute:
            "avg_runtime": float(self.runtime_reward) if self.runtime_reward_vector else 0.0,
            "avg_waiting": float(self.waiting_reward) if self.waiting_reward_vector else 0.0,
            "avg_energy_per_job": (self.total_energy / max(completed, 1)),
            "battery_soc": float(self.battery.current / self.battery.capacity),
        }
        # Also expand per-core utilization into scalar columns for CSV:
        metrics.update({f"core{i}_util": float(u) for i, u in enumerate(utilpercore)})
        if mode == 'human':
            print("=" * 48)
            print(f"Makespan:                 {metrics['makespan']:.3f}")
            print(f"Completed / On-time:      {completed} / {finished_on_time}")
            print(f"Missed / Dropped:         {missed} / {dropped}")
            print(f"slack_counter:            {slack_counter}")
            print(f"slack_delay:              {total_delay_slack}")
            print(f"Miss rate (completed):    {metrics['deadline_miss_rate_completed']:.3%}")
            print(f"Miss rate (+drops):       {metrics['deadline_miss_rate_including_drops']:.3%}")
            print(f"Throughput:               {metrics['throughput_tasks_per_time']:.6f} tasks/time")
            print(f"Total energy:             {metrics['total_energy_consumption']:.6f}")
            print(f"Total core busy time:     {metrics['total_core_busy_time']:.6f}")
            print(f"System utilization:       {metrics['system_utilization']:.2%}")
            # Pretty per-core line (array can't be formatted with :.2%, so do item-by-item)
            per_core_line = " ".join(f"C{i+1}={u:.2%}" for i, u in enumerate(utilpercore))
            print(f"Cores utilization:        {per_core_line}")
            print(f"Avg runtime / waiting:    {metrics['avg_runtime']:.6f} / {metrics['avg_waiting']:.6f}")
            print(f"Avg energy per job:       {metrics['avg_energy_per_job']:.6f}")
            print(f"Battery SoC:              {metrics['battery_soc']:.2%}")
            print (f'total exex {self.total_execution}')
            print("=" * 48)
        return metrics



    def _norm_time(self, x: float) -> float:
        return float(max(0.0, min(1.0, x / (self.max_deadline+ 1) )))

    def _norm_energy(self, e: float) -> float:
        return float(max(0.0, min(1.0, e / float(self.battery.capacity))))

    def get_time(self) -> int:
        return int(self.t)

    def get_max_delay_time(self, i, m) -> int:
        # print (m, len(self.activeQueue))
        # print(self.activeQueue)
        if self.activeQueue and self.check_assign(i, m) :
            job = self.activeQueue[int(m)]
            # if job.naughty_task :
            #     # print ("naughty reach here ! ")
            #     job.naughty_task = False
            #     job.rt_est[int(i)] -= job.task_delay
            #     job.task_delay = 0
            # print(f'before fucked : {job.deadline}, {job.rt_est[int(i)]},{job.task_name}, {job.task_delay} and its damn chosen core ! : {i+1}')
            max_delay = job.deadline - job.rt_est[int(i)]
            if job.rt_est[int(i)] > 100000000000000000 :
                return  0
            return max_delay
        else :
            return  0
