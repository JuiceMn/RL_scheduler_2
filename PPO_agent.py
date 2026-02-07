import os

import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
import numpy as np
import torch
import gymnasium as gym
# import time
# import matplotlib.pyplot as plt
deque = __import__('collections').deque
from typing import List, Tuple, Optional

# 1) Action-space helpers
from actions import ActionDecoder
import os, csv
from scheduler import Scheduler

# -----------------------------------------
# 1) Gym wrapper around Scheduler
# -----------------------------------------
class SchedulerEnv(gym.Env):
    def __init__(self, scheduler: Scheduler, action_decoder: ActionDecoder,
                 render_mode: Optional[str] = "human",
                 log_csv_path: Optional[str] = None,
                 fieldnames: Optional[list[str]] = None
                 ):
        super().__init__()
        self.scheduler = scheduler
        self.ActionDecoder = action_decoder
        self.render_mode = render_mode
        self.log_csv_path = log_csv_path
        self.fieldnames = fieldnames

        # composite action space = (assign, delay)
        # self.action_space = gym.spaces.Tuple((
        #     self.ActionDecoder.assign_space,
        #     self.ActionDecoder.delay_space
        # ))
        # self.assign_space = gym.spaces.Discrete(self.scheduler.M * self.scheduler.I + 1)
        # self.delay_space = gym.spaces.Discrete(20)

        dim = len(self.scheduler.build_state())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # self.scheduler.arrival.reset(seed)
        st = self.scheduler.reset()
        return np.array(st, dtype=np.float32), {}
##############
    def get_valid_mask(self) -> np.ndarray:
        """Boolean mask [A] of legal assign actions at the current state."""
        m = self.ActionDecoder.valid_assign_mask(self.scheduler)  # torch.bool or np
        # print(m)
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)

        return m.astype(bool)
####
    @staticmethod
    def _append_metrics_csv(path: str, row: dict, fieldnames: list[str]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            # keep only known keys (stable column order)
            w.writerow({k: row.get(k, "") for k in fieldnames})
    def render(self):
        """
        - Uses self.render_mode ('human' prints; 'ansi' returns a string).
        - If self.log_csv_path is set, appends metrics row to CSV (fieldnames fixed or inferred).
        - Returns None for 'human' (Gym convention) and a string for 'ansi'.
        """
        mode = self.render_mode or "human"

        # scheduler.render prints in 'human' and returns a dict of metrics in all modes
        metrics = self.scheduler.render(mode=mode)

        # CSV logging (once per call)
        if self.log_csv_path is not None:
            if self.fieldnames is None:
                # infer stable column order from first metrics dict
                self.fieldnames = list(metrics.keys())
            self._append_metrics_csv(self.log_csv_path, metrics, self.fieldnames)

        if mode == "ansi":
            # return a compact, single-string snapshot
            s = (
                f"Makespan={metrics.get('makespan', 0):.3f}, "
                f"Completed={metrics.get('completed', 0)}, "
                f"OnTime={metrics.get('finished_on_time', 0)}, "
                f"MissRate={metrics.get('deadline_miss_rate_completed', 0):.3%}, "
                f"Util={metrics.get('system_utilization', 0):.2%}, "
                f"Throughput={metrics.get('throughput_tasks_per_time', 0):.6f}, "
                f"Energy={metrics.get('total_energy_consumption', 0):.6f}"
            )
            return s

    def step(self, action: Tuple[int, int]
             ): #-> Tuple[np.ndarray, float, bool, dict]:
        assign_int, delay_int = action
        # print(assign_int, delay_int)

        # decode your two sub-actions
        cmd, core, slot = self.ActionDecoder.decode_assign(assign_int)
        # (delay_int is just the number of ticks to hold)
        # print(f'time:{self.scheduler.t}')

        state, reward, done, garbage, info = self.scheduler.step(
            (cmd, core, slot),
            delay_int
        )

        # print(np.array(state).shape)
        return np.array(state, dtype=np.float32), reward, done, False, info

class Buffer():
    def __init__(self):
        self.buffer_num = 0
        self.states = []
        self.actions = []
        self.masks = []
        self.log_probs = []
        self.Returns = []
        self.advantages = []

    def clear_buffer(self):
        self.buffer_num = 0
        self.states = []
        self.actions = []
        self.masks = []
        self.log_probs = []
        self.Returns = []
        self.advantages = []

    def store_buffer(self, state, mask, action, log_prob, Return, advantage, nums):
        self.buffer_num = self.buffer_num + nums
        self.states.extend(state)
        self.masks.extend(mask)
        self.actions.extend(action)
        self.log_probs.extend(log_prob)
        self.Returns.extend(Return)
        self.advantages.extend(advantage)


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, device=None):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            # print(self.masks, "fake", masks.shape)
            # print(logits, "logits")
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            # print(self.masks.shape)
            # print(logits.shape)
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)



# class ActorNet(nn.Module):
#     def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3, num_inputs4, featureNum4
#                  , action_dim):
#         super(ActorNet, self).__init__()
#         self.d_model = 128
#
#         self.num_inputs1 = num_inputs1
#         self.featureNum1 = featureNum1
#         self.num_inputs2 = num_inputs2
#         self.featureNum2 = featureNum2
#         self.num_inputs3 = num_inputs3
#         self.featureNum3 = featureNum3
#         self.num_inputs4 = num_inputs4
#         self.featureNum4 = featureNum4
#         self.action_dim = action_dim
#
#         self.JobEncoder = nn.Sequential(
#             nn.Linear(self.featureNum1, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.d_model),
#             nn.ReLU(),
#         )
#
#         self.decoder1 = nn.Sequential(
#             nn.Linear(self.d_model, 64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.ReLU(),
#             nn.Linear(16, self.action_dim), #fined better way to put action dim for better proformance !
#         )
#
#     def forward(self, x):
#         job = x[:, :self.num_inputs1, :self.featureNum1]
#         job = self.JobEncoder(job)
#         con = job
#         con =  self.decoder1(con)
#
#         logits = con.squeeze(dim=-1)
#
#         return logits


class ActorNet(nn.Module):
    """
    Outputs joint action logits for (job_i -> core_j) plus HOLD:
      logits shape: (B, Q*C + 1) = (B, 82) for Q=C=9
    """
    def __init__(self,
                 # row counts
                 num_wait, num_run, num_green, num_core,
                 # feature counts
                 feat_wait, feat_run, feat_green, feat_core,
                 d_model=128, pair_hidden=64, ctx_dim=128):
        super().__init__()
        self.Q = num_wait
        self.M = num_run
        self.C = num_core
        self.G = num_green

        # --- per-row encoders (shared MLP applied to last dim) ---
        def enc(f_in):
            return nn.Sequential(
                nn.Linear(f_in, 64), nn.ReLU(),
                nn.Linear(64, d_model), nn.ReLU()
            )
        self.JobEnc   = enc(feat_wait)    # (B,Q,feat_wait)  -> (B,Q,d)
        self.CoreEnc  = enc(feat_core)    # (B,C,feat_core)  -> (B,C,d)
        self.RunEnc   = enc(feat_run)     # (B,M,feat_run)   -> (B,M,d)
        self.GreenEnc = enc(feat_green)   # (B,G,feat_green) -> (B,G,d)

        # --- context from non-action windows (mean-pool) ---
        self.CtxFuse = nn.Sequential(
            nn.Linear(2 * d_model, ctx_dim), nn.ReLU()  # run + green pooled
        )

        # --- project to pair space and score every (i,j) ---
        self.Jproj = nn.Linear(d_model, pair_hidden)    # job row -> h
        self.Cproj = nn.Linear(d_model, pair_hidden)    # core row -> h
        self.CtxProj = nn.Linear(ctx_dim, pair_hidden)  # broadcasted ctx -> h
        self.PairScore = nn.Linear(pair_hidden, 1)      # h -> scalar

        # --- HOLD head (uses only context) ---
        self.HoldHead = nn.Linear(ctx_dim, 1)

    def forward(self, x):
        """
        x layout on rows (slots): [0:Q)=waiting, [Q:Q+M)=running,
                                  [Q+M:Q+M+C)=cores, [end)=green
        last dim is the max feature width; each slice uses its own :feat
        """
        B = x.size(0)
        Q = self.Q
        M = self.M
        C = self.C

        # slice windows with their feature counts
        jobs   = x[:, 0:Q,                     :self.JobEnc[0].in_features]    # (B,Q,Fw)
        runs   = x[:, Q:Q+M,                   :self.RunEnc[0].in_features]    # (B,M,Fr)
        greens = x[:, Q+M:Q+M+self.G,          :self.GreenEnc[0].in_features]   # (B,C,Fc)
        cores =  x[:, Q+M+self.G:Q+M+self.G+C, :self.CoreEnc[0].in_features ]  # (B,G,Fg)

        # encode rows
        J = self.JobEnc(jobs)        # (B,Q,d)
        R = self.RunEnc(runs)        # (B,M,d)
        Cw = self.CoreEnc(cores)     # (B,C,d)
        G = self.GreenEnc(greens)    # (B,G,d)

        # pooled context from run & green
        run_pool   = R.mean(dim=1)                # (B,d)
        green_pool = G.mean(dim=1)                # (B,d)   (G=1 -> identity)
        ctx = self.CtxFuse(torch.cat([run_pool, green_pool], dim=-1))  # (B,ctx_dim)

        # project into a shared hidden space
        Jh = self.Jproj(J)                        # (B,Q,h)
        Ch = self.Cproj(Cw)                       # (B,C,h)
        Ct = self.CtxProj(ctx).unsqueeze(1).unsqueeze(1)  # (B,1,1,h)

        # broadcast add: score every (job i, core j)
        # result S: (B, Q, C, h)
        S = torch.tanh(Jh.unsqueeze(2) + Ch.unsqueeze(1) + Ct)
        S = self.PairScore(S).squeeze(-1)         # (B, Q, C)
        #####
        # print("x shape:", x.shape)  # expect (B, 28, 24)
        # print("Q,M,C,G:", self.Q, self.M, self.C, self.G)
        # print("Fw,Fr,Fc,Fg:",
        #       self.JobEnc[0].in_features,
        #       self.RunEnc[0].in_features,
        #       self.CoreEnc[0].in_features,
        #       self.GreenEnc[0].in_features)
        #
        # print("jobs rows:", x[:, 0:self.Q].shape[1])
        # print("runs rows:", x[:, self.Q:self.Q + self.M].shape[1])
        # print("cores rows:", x[:, self.Q + self.M + self.G:self.Q + self.M + self.G + self.C].shape[1])
        # print("greens rows:", x[:, self.Q + self.M:self.Q + self.M + self.G].shape[1])
        #
        # # after encoders:
        # print("J,R,Cw,G shapes:", J.shape, R.shape, Cw.shape, G.shape)
        #
        # # before reshape:
        # print("S pre-score shape should be (B,Q,C,h):", (Jh.unsqueeze(2) + Ch.unsqueeze(1) + Ct).shape)
        # print("S post-score shape (B,Q,C):", S.shape, "numel:", S.numel())
        #####
        pair_logits = S.reshape(B, Q * C)         # (B, 81)

        hold_logit = self.HoldHead(ctx)           # (B, 1)

        logits = torch.cat([pair_logits, hold_logit], dim=-1)  # (B, 82)
        return logits


class CriticNet(nn.Module):

    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3, featureNum4, num_inputs4):
        super(CriticNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3
        self.num_inputs4 = num_inputs4
        self.featureNum4 = featureNum4

        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.GreenEncoder = nn.Sequential(
            nn.Linear(self.featureNum3, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(self.featureNum2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )
        self.CoreEncoder = nn.Sequential(   # we give for cores feature part ! btw!
            nn.Linear(self.featureNum4, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear((self.num_inputs1 + self.num_inputs2 + self.num_inputs3 + self.num_inputs4) * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        run = x[:, self.num_inputs1:self.num_inputs1 + self.num_inputs2, :self.featureNum2]
        green = x[:, self.num_inputs1 + self.num_inputs2:self.num_inputs1 + self.num_inputs2 + self.num_inputs3,
                :self.featureNum3]
        core = x[:, self.num_inputs1 + self.num_inputs2 + self.num_inputs3 : self.num_inputs1 + self.num_inputs2 + self.num_inputs3 + self.num_inputs4,
               :self.featureNum4]
        green = self.GreenEncoder(green)
        job = self.JobEncoder(job)
        run = self.RunningJobEncoder(run)
        # print (core.shape, "core")
        # print (job.shape, "job")
        # print (green.shape, "green")
        # print (run.shape, "running")
        core = self.CoreEncoder(core)
        con = torch.cat([job, run, green, core], dim=1)

        con = self.hidden(con)
        con = self.flatten(con)
        value = self.out(con)
        return value


class PPO():
    def __init__(self, batch_size=681, inputNum_size=[20, 9, 1, 9], featureNum_size=[24, 8, 3, 5],
                 device='cpu'):
        super(PPO, self).__init__()
        ## how this part is made ? !
        self.num_inputs1 = inputNum_size[0] #waiting
        self.num_inputs2 = inputNum_size[1] #running
        self.num_inputs3 = inputNum_size[2] #green
        self.num_inputs4 = inputNum_size[3] #core

        self.featureNum1 = featureNum_size[0]
        self.featureNum2 = featureNum_size[1]
        self.featureNum3 = featureNum_size[2]
        self.featureNum4 = featureNum_size[3]

        self.device = device
        self.actor_net = ActorNet(
            self.num_inputs1, self.num_inputs2, self.num_inputs3, self.num_inputs4, self.featureNum1,
            self.featureNum2, self.featureNum3, self.featureNum4).to(self.device)
        self.critic_net = CriticNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3, self.featureNum4, self.num_inputs4).to(self.device)
        self.batch_size = batch_size
        self.gamma = 1
        self.lam = 0.97

        self.states = []
        self.log_probs = []
        self.rewards_seq = []
        self.actions = []
        self.values = []
        self.masks = []
        self.entropys = []
        self.buffer = Buffer()

        self.ppo_update_time = 8
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.entropy_coefficient = 0

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=0.0001,eps=1e-6)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=0.0005,eps=1e-6)

    def act(self, state, mask):
        logits = self.actor_net(state)

        value = self.critic_net(state)
        dist_bin = CategoricalMasked(logits=logits, masks=mask,device=self.device)
        id = dist_bin.sample()

        log_prob = dist_bin.log_prob(id)
        return id, log_prob, value

    def act1(self, state, mask, action):
        logits = self.actor_net(state)

        dist_bin = CategoricalMasked(logits=logits, masks=mask,device=self.device)
        log_prob = dist_bin.log_prob(action)
        entropy = dist_bin.entropy()
        return log_prob, entropy

    def normalize(self, advantages):
        nor_advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-9)
        return nor_advantages

    def remember(self, state, value, log_prob, action, reward, mask, device):
        self.rewards_seq.append(reward)
        self.states.append(state.to("cpu"))
        self.log_probs.append(log_prob.to("cpu"))
        self.values.append(value.to("cpu"))
        self.actions.append(action.to("cpu"))
        self.masks.append(mask.to("cpu"))

    def clear_memory(self):
        self.rewards_seq = []
        self.states = []
        self.log_probs = []
        self.values = []
        self.actions = []
        self.masks = []

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        rews = np.append(np.array(self.rewards_seq), last_val)
        values = torch.cat(self.values, dim=0)
        values = values.squeeze(dim=-1)
        vals = np.append(np.array(values.cpu()), last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        ret = self.discount_cumsum(rews, self.gamma)[:-1]
        # ret=adv+vals[:-1]
        return adv, ret

    def storeIntoBuffter(self, reward):
        advantages, returns = self.finish_path(reward)
        returns = returns.tolist()
        advantages = advantages.tolist()

        self.buffer.store_buffer(self.states, self.masks, self.actions, self.log_probs, returns, advantages,
                                 len(self.states))

    def compute_value_loss(self, states, returns):
        state_values = self.critic_net(states)
        state_values = torch.squeeze(state_values, dim=1)

        # Calculate value loss using F.mse_loss
        value_loss = F.mse_loss(state_values, returns)
        return value_loss

    def compute_actor_loss(self,
                           states,
                           masks,
                           actions,
                           advantages,
                           old_log_probs
                           ):

        log_probs, entropy = self.act1(states, masks, actions)
        # Compute the policy loss
        logratio = log_probs - old_log_probs
        ratio = torch.exp(logratio)

        surr1 = ratio * advantages
        clip_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        surr2 = clip_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))  # MAX->MIN descent
        entropy_loss = torch.mean(entropy)

        total_loss = policy_loss - self.entropy_coefficient * entropy_loss

        return total_loss, policy_loss, entropy_loss

    def train(self):
        states = torch.cat(self.buffer.states, dim=0)
        masks = torch.cat(self.buffer.masks, dim=0)
        actions = torch.cat(self.buffer.actions, dim=0)
        log_probs = torch.cat(self.buffer.log_probs, dim=0)
        returns = torch.tensor(self.buffer.Returns)
        advantages = torch.tensor(self.buffer.advantages)
        advantages = self.normalize(advantages)
        # print(self.batch_size)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), self.batch_size, False):
                index_tensor = torch.tensor(index)
                sampled_states = torch.index_select(states, dim=0, index=index_tensor).to(self.device)
                sampled_masks = torch.index_select(masks, dim=0, index=index_tensor).to(self.device)
                sampled_actions = torch.index_select(actions, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs = torch.index_select(log_probs, dim=0, index=index_tensor).to(self.device)
                sampled_returns = torch.index_select(returns, dim=0, index=index_tensor).to(self.device)
                sampled_advantages = torch.index_select(advantages, dim=0, index=index_tensor).to(self.device)
                action_loss, polic_loss, entropy_loss = self.compute_actor_loss(sampled_states, sampled_masks,
                                                                                sampled_actions, sampled_advantages,
                                                                                sampled_log_probs)
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)

                self.actor_optimizer.zero_grad()
                self.critic_net_optimizer.zero_grad()

                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

    def save_using_model_name(self, model_name_path):
        if not os.path.exists(model_name_path):
            # 如果目录不存在，则创建它 wtf nigga !!!!
            os.makedirs(model_name_path)
        torch.save(self.actor_net.state_dict(), model_name_path + "actor.pkl")
        torch.save(self.critic_net.state_dict(),
                   model_name_path + "critic.pkl")

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(
            torch.load(model_name_path + "actor.pkl"))
        self.critic_net.load_state_dict(
            torch.load(model_name_path + "critic.pkl"))

    def eval_action(self, o, mask, greedy : bool = False ):
        with torch.no_grad():
            if torch.is_tensor(o):
                # this need to be checked !
                o = o.reshape(1, 20 + 9 + 1 + 9, 24)
                state = o.to(self.device, dtype=torch.float32)
            else:
                o = np.asarray(o).reshape(1, 20 + 9 + 1 + 9, 24)
                state = torch.as_tensor(o, dtype=torch.float32, device=self.device)

                # *** FIX: keep mask as bool tensor on the right device ***
            if torch.is_tensor(mask):
                mask = mask.to(self.device, dtype=torch.bool)
            else:
                mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

            logits = self.actor_net(state)
            if greedy :
                value = self.critic_net(state)
                masked = logits.masked_fill(~mask, float("-inf"))
                ac = masked.argmax(dim=-1)  # (1,)
            else :
                value = self.critic_net(state)
                dist_bin = CategoricalMasked(logits=logits, masks=mask, device=self.device)
                ac = dist_bin.sample()                      ## mask should to be cheacked !

        return ac.item()




def train(env):
    seed = 0
    epochs = 100
    traj_num = 100
    # env = SchedulerEnv(scheduler, action_decoder)
    # env.seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # inputNum_size   = [waiting_queue, running_queue, green, core_queue]
    inputNum_size     = [20, 9, 1, 9]
    # featureNum_size = [JOB_FEATURES, RUN_FEATURES, GREEN_FEATURE, CORE_FEATURES]
    featureNum_size   = [24, 8, 3, 5]
    ep_len_int = 0
    ppo = PPO(batch_size=ep_len_int, inputNum_size=inputNum_size,
              featureNum_size=featureNum_size, device=device)
    for epoch in range(epochs):
        print(epoch)
        o, _ = env.reset()
        print(o.shape)
        r = 0
        d = False
        ep_ret = 0
        ep_len = 0
        #### obs, _ = env.reset() need to be this ! what the shit ?! [fixed]
        t = 0
        run_time    = 0
        waiting     = 0
        slack       = 0
        epoch_reward= 0
        while True:

            with torch.no_grad():
                # o = o.reshape(1, core_queue + green + wating_queue + running_queue, JOB_FEATURES)
                o = o.reshape(1, 20 + 9 + 1 + 9, 24)
                state = torch.FloatTensor(o).to(device)
                mask = env.get_valid_mask() ## for now just assign_task ! next we need delay as well !
                mask = torch.tensor(np.stack(mask), dtype=torch.bool, device=device)

                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)  # (1, 82)
                # print(mask.shape, "org")
                # pack that 3 up code into 1 code next time !
                # if mask.any() :
                #     print(mask)
                ind, log_prob, value = ppo.act(state, mask)
            ppo.remember(state, value, log_prob, ind, r, mask, device)

            o, r, d, truncated, infos = env.step((ind.item(),0))
            ep_ret += r
            ep_len += 1
            ###############
            if ep_ret != 0 :
                print(ep_ret, ep_len, 'time:',env.scheduler.t, "nice_task:",env.scheduler.finish_on_time,
                        "all execution : ", sum(env.scheduler.runtime_reward_vector), "all energy", sum(env.scheduler.energy_reward_vector),
                      "total wipe:", env.scheduler.total_wipe,
                      "total miss :", env.scheduler.active_queue_miss_counter,
                      "all task :",   env.scheduler.task_name_finished  )
            ################
            # show_ret += r2
            # sjf += sjf_t
            # f1 += f1_t
            total_r, rt, wt, st = env.scheduler._compute_reward()

            run_time                 += rt
            waiting                   += wt
            slack                    += st
            epoch_reward             += r

            if d:
                t += 1
                ppo.storeIntoBuffter(r)
                ppo.clear_memory()
                o, _ = env.reset()
                r = 0
                d = False
                ep_len_int = ep_len
                ppo.batch_size = ep_len_int
                # print(ep_len_int)
                ep_ret = 0
                ep_len = 0
                print(f'{t} : traj_num')
                if t >= traj_num:
                    break

        ppo.train()
        with open('MaskablePPO_' + "10landa" + '.csv', mode='a',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow([float(epoch_reward / traj_num),float(run_time / traj_num),float(waiting / traj_num),float(slack / traj_num)])
        ppo.buffer.clear_buffer()

    ppo.save_using_model_name('Each_10' + '/MaskablePPO')