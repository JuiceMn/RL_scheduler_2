
from cores_and_tasks_batt import Battery, Task, SolarChargeRate
from arrivalprocess_taskfactory import PoissonArrivalProcess, build_kernel_param_map, load_arrival_trace
from actions import ActionDecoder
from scheduler import Scheduler
# from RL_agent_one_head_paralell_GAE import SchedulerEnv, A2CAgent, A2CTrainer, moving_avg, loss_per_episode
from GHI_On_Hottest_Day import irr_profile
from kernel_specs import kernel_specs
from cores import cores
from gymnasium.vector import AsyncVectorEnv
# from PPO_MARL_agent import train, SchedulerEnv
import matplotlib.pyplot as plt
from PPO_agent import train, SchedulerEnv


def main():
    # 1) task_factory
    # types, param_map = build_kernel_param_map(cores, kernel_specs)

    # arrival = PoissonArrivalProcess(
    #     rate          = 0.5,
    #     types         = types,
    #     param_map     = param_map,
    #     mean_deadline = 50,
    #     std_deadline  = 10,
    #     seed          = 42,
    # )
    arrival, total_index = load_arrival_trace("arrival_trace.csv",
                                 cores=cores,
                                 kernel_specs=kernel_specs,
                                 use_absolute_deadline=True)

    # 2) Solar + Battery
    solar = SolarChargeRate(alpha=0.18, area=1.2, irr_profile=irr_profile)
    battery = Battery(capacity= 2_000_000.0, rate_process=solar,
                      initial_charge =2000000.0 , begining_time=6*3600*1000)

    # 3) Scheduler
    M_window, H, O_max, L_max, T_window = 20, 900000, 10, 5, 100
    feat_len = Task.feature_length(len(cores))
    scheduler = Scheduler(cores, battery, arrival,
                          M_window, H, O_max, L_max, T_window, feat_len, total_index)

    action_decoder = ActionDecoder(M_window, len(cores), D_max=100)
    env = SchedulerEnv(scheduler, action_decoder)
    train (env)
    # 4) Gym wrapper + action‐decoder


    # def make_env_fn(seed: int):
    #     def _thunk():
    #
    #         env = SchedulerEnv(scheduler, action_decoder)
    #         env.reset(seed=seed)
    #         return env
    #
    #     return _thunk
    #################

    #################
    #################################
    # N_ENVS = 16
    # seeds = [10_000 + i for i in range(N_ENVS)]
    # env_fns = [make_env_fn(s) for s in seeds]
    # venv = AsyncVectorEnv(env_fns)
    #################################
    # # 5) Agent & Trainer
    # state_dim  = env.single_observation_space.shape[0]
    # # print(state_dim)
    # assign_dim = action_decoder.assign_space.n
    # # delay_dim  = action_decoder.delay_space.n
    #
    # agent = A2CAgent(
    #     state_dim  = state_dim ,
    #     action_dim = assign_dim,
    # )
    # rollout_T = 32 ## need to check
    # trainer = A2CTrainer(agent, rollout_T, venv, action_decoder)
    #
    # # 6) Fire off training
    # logs = trainer.training_batch()
    # plt.figure()
    # plt.plot(logs["episode_rewards"], label="Episode reward")
    # ma = moving_avg(logs["episode_rewards"], k=50)
    # if len(ma): plt.plot(range(50 - 1, 50 - 1 + len(ma)), ma, label="Moving avg (50)")
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.legend()
    # plt.title("Episode Rewards")
    # plt.tight_layout()
    # plt.show()
    #
    # # Loss (approx per episode)
    # loss_ep = loss_per_episode(logs["losses"], episodes_finished_per_update=None)
    # plt.figure()
    # plt.plot(loss_ep["total"], label="Loss per episode (expanded)")
    # ma = moving_avg(loss_ep["total"], k=100)
    # if len(ma): plt.plot(range(100 - 1, 100 - 1 + len(ma)), ma, label="Moving avg (100)")
    # plt.xlabel("Episode")
    # plt.ylabel("Total loss")
    # plt.legend()
    # plt.title("Loss per Episode")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
