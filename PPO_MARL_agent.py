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
import math
from torch.distributions import Beta

# import time
# import matplotlib.pyplot as plt
deque = __import__('collections').deque
from typing import List, Tuple, Optional

# 1) Action-space helpers
from actions import ActionDecoder


from scheduler import Scheduler

# -----------------------------------------
# 1) Gym wrapper around Scheduler
# -----------------------------------------
class SchedulerEnv(gym.Env):
    def __init__(self, scheduler: Scheduler, action_decoder: ActionDecoder):
        super().__init__()
        self.scheduler = scheduler
        self.ActionDecoder = action_decoder

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
    def get_valid_assign_mask(self) :
        """Boolean mask [A] of legal assign actions at the current state."""
        m = self.ActionDecoder.valid_assign_mask(self.scheduler)  # torch.bool or np
        # n = self.ActionDecoder.valid_delay_mask(self.scheduler)
        # print(m)
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        # n = n.detach().cpu().numpy() if hasattr(n, "detach") else np.asarray(n)
        return m.astype(bool) #n.astype(bool)
    def get_valid_delay_mask(self, assign) :
        n = self.ActionDecoder.max_delay_bound(self.scheduler, assign)
        n = n.detach().cpu().numpy() if hasattr(n, "detach") else np.asarray(n)
        return n
    def decode_assign (self, assign):
        decode = self.ActionDecoder.decode_assign(assign)
        return decode
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
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.Returns = []
        self.advantages = []

    def clear_buffer(self):
        self.buffer_num = 0
        self.states = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.Returns = []
        self.advantages = []

    def store_buffer(self, state, mask1, mask2, action1, action2, log_prob1, log_prob2, Return, advantage,nums):
        self.buffer_num = self.buffer_num + nums
        self.states.extend(state)
        self.masks1.extend(mask1)
        self.masks2.extend(mask2)
        self.actions1.extend(action1)
        self.actions2.extend(action2)
        self.log_probs1.extend(log_prob1)
        self.log_probs2.extend(log_prob2)
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
    Head 1: logits for (job_i -> core_j) plus HOLD  => (B, Q*C + 1)
    Head 2: continuous delay fraction u in (0,1) conditioned on (state + selected pair)
            modeled as Beta(alpha, beta). You then map to integer delay using max_delay:
                delay_int = round(u * max_delay)

    NOTE: This module returns logits/params only. No environment-specific max_delay inside.
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
        self.G = num_green
        self.C = num_core
        self.d_model = d_model
        self.pair_hidden = pair_hidden

        # --- per-row encoders (shared MLP per row-type) ---
        def enc(f_in):
            return nn.Sequential(
                nn.Linear(f_in, 64), nn.ReLU(),
                nn.Linear(64, d_model), nn.ReLU()
            )
        self.JobEnc   = enc(feat_wait)    # (B,Q,feat_wait)  -> (B,Q,d)
        self.RunEnc   = enc(feat_run)     # (B,M,feat_run)   -> (B,M,d)
        self.GreenEnc = enc(feat_green)   # (B,G,feat_green) -> (B,G,d)
        self.CoreEnc  = enc(feat_core)    # (B,C,feat_core)  -> (B,C,d)

        # --- context for pair head (mean-pool over runs & green) ---
        self.CtxFuse = nn.Sequential(
            nn.Linear(2 * d_model, ctx_dim), nn.ReLU()
        )

        # --- project to pair space and score every (i,j) ---
        self.Jproj = nn.Linear(d_model, pair_hidden)
        self.Cproj = nn.Linear(d_model, pair_hidden)
        self.CtxProj = nn.Linear(ctx_dim, pair_hidden)
        self.PairScore = nn.Linear(pair_hidden, 1)

        # --- HOLD head (uses mean-pooled context) ---
        self.HoldHead = nn.Linear(ctx_dim, 1)

        # ===========================
        # Delay head: state + assignment -> Beta(alpha,beta)
        # ===========================
        self.TypeEmb = nn.Embedding(4, d_model)

        # Single-query pooling (Set-style attention)
        self.AttnWk  = nn.Linear(d_model, d_model, bias=False)
        self.attn_q  = nn.Parameter(torch.randn(d_model))

        # project pair interaction (h) up to d for concatenation
        self.PairCtxToD = nn.Linear(pair_hidden, d_model)
        nn.init.xavier_uniform_(self.PairCtxToD.weight)

        # learned stand-ins for HOLD (no specific job/core)
        self.HoldJobEmb  = nn.Parameter(torch.zeros(d_model))
        self.HoldCoreEmb = nn.Parameter(torch.zeros(d_model))
        self.HoldPairCtx = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.HoldJobEmb,  std=0.02)
        nn.init.normal_(self.HoldCoreEmb, std=0.02)
        nn.init.normal_(self.HoldPairCtx, std=0.02)

        # Delay head outputs 2 params (alpha_raw, beta_raw) -> alpha,beta via softplus
        self.DelayHead = nn.Sequential(
            nn.Linear(4 * d_model, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    # ----------------------------
    # Robust conversion helpers
    # ----------------------------
    def _to_tensor(self, x, *, device=None, dtype=None):
        """Convert numpy/scalar/torch -> torch tensor on device."""
        if torch.is_tensor(x):
            t = x
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            if device is not None and t.device != device:
                t = t.to(device)
            return t

        # numpy / scalar / list
        t = torch.as_tensor(x, dtype=dtype if dtype is not None else None)
        if device is not None:
            t = t.to(device)
        return t

    def _ensure_state(self, x):
        """State must be float tensor on module device."""
        device = next(self.parameters()).device
        x = self._to_tensor(x, device=device, dtype=torch.float32)
        return x

    def _ensure_pair_action(self, pair_action, B, device):
        """
        pair_action must be Long tensor shape (B,).
        Accepts numpy array / scalar / torch.
        """
        pa = self._to_tensor(pair_action, device=device, dtype=torch.long).view(-1)

        # If scalar provided and B>1, broadcast
        if pa.numel() == 1 and B > 1:
            pa = pa.expand(B)

        # If wrong size, try squeeze common (B,1) -> (B,)
        if pa.numel() != B:
            # last resort: allow (B,) only
            raise ValueError(f"pair_action size mismatch: got {pa.shape}, expected ({B},)")

        return pa

    def _ensure_max_delay(self, max_delay, B, device):
        """max_delay must be Long tensor shape (B,)."""
        md = self._to_tensor(max_delay, device=device, dtype=torch.long).view(-1)
        if md.numel() == 1 and B > 1:
            md = md.expand(B)
        if md.numel() != B:
            raise ValueError(f"max_delay size mismatch: got {md.shape}, expected ({B},)")
        return md

    # ---------- helpers ----------
    def _attn_pool(self, tokens, mask=None):
        """
        tokens: (B, N, d)
        mask:   (B, N) bool, True means keep; False masked out
        returns pooled: (B, d)
        """
        B, N, d = tokens.shape
        K = self.AttnWk(tokens)  # (B,N,d)
        scores = torch.einsum('bnd,d->bn', K, self.attn_q) / math.sqrt(d)
        if mask is not None:
            # ensure mask tensor
            mask = self._to_tensor(mask, device=tokens.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask, float('-inf'))
        w = F.softmax(scores, dim=1)
        pooled = torch.einsum('bn,bnd->bd', w, tokens)
        return pooled

    def _encode_rows(self, x):
        """
        Returns encoded slices + context + tokens for attention, and pair hidden tensor for all (i,j).
        """
        x = self._ensure_state(x)
        B = x.size(0)
        Q, M, G, C = self.Q, self.M, self.G, self.C

        jobs   = x[:, 0:Q,               :self.JobEnc[0].in_features]
        runs   = x[:, Q:Q+M,             :self.RunEnc[0].in_features]
        greens = x[:, Q+M:Q+M+G,         :self.GreenEnc[0].in_features]
        cores  = x[:, Q+M+G:Q+M+G+C,     :self.CoreEnc[0].in_features]

        J  = self.JobEnc(jobs)       # (B,Q,d)
        R  = self.RunEnc(runs)       # (B,M,d)
        Gv = self.GreenEnc(greens)   # (B,G,d)
        Cw = self.CoreEnc(cores)     # (B,C,d)

        run_pool   = R.mean(dim=1) if M > 0 else torch.zeros(B, J.size(-1), device=J.device, dtype=J.dtype)
        green_pool = Gv.mean(dim=1) if G > 0 else torch.zeros(B, J.size(-1), device=J.device, dtype=J.dtype)
        ctx = self.CtxFuse(torch.cat([run_pool, green_pool], dim=-1))  # (B,ctx_dim)

        Jh = self.Jproj(J)                               # (B,Q,h)
        Ch = self.Cproj(Cw)                              # (B,C,h)
        Ct = self.CtxProj(ctx).unsqueeze(1).unsqueeze(1) # (B,1,1,h)
        H = torch.tanh(Jh.unsqueeze(2) + Ch.unsqueeze(1) + Ct)         # (B,Q,C,h)

        tok = []
        if Q > 0: tok.append(J  + self.TypeEmb.weight[0])
        if M > 0: tok.append(R  + self.TypeEmb.weight[1])
        if C > 0: tok.append(Cw + self.TypeEmb.weight[2])
        if G > 0: tok.append(Gv + self.TypeEmb.weight[3])
        tokens = torch.cat(tok, dim=1)  # (B, Q+M+C+G, d)

        return J, R, Gv, Cw, ctx, H, tokens

    def _pair_logits_from_H(self, H):
        S = self.PairScore(H).squeeze(-1)  # (B,Q,C)
        return S.reshape(S.size(0), -1)    # (B, Q*C)

    # ---------- pair head ----------
    def forward(self, x):
        """
        Returns pair head logits only: (B, Q*C+1).
        """
        _, _, _, _, ctx, H, _ = self._encode_rows(x)
        pair_logits = self._pair_logits_from_H(H)        # (B, Q*C)
        hold_logit  = self.HoldHead(ctx)                 # (B, 1)
        return torch.cat([pair_logits, hold_logit], dim=-1)

    # ---------- delay head (continuous fraction) ----------
    def delay_params_with_action(self, x, pair_action, attn_mask=None, eps=1e-4):
        """
        Returns (alpha, beta, is_hold).
        alpha,beta: (B,) positive
        """
        J, _, _, Cw, _, H, tokens = self._encode_rows(x)
        B = tokens.size(0)
        device = tokens.device

        # FIX: pair_action may be numpy => convert to torch.LongTensor
        pair_action = self._ensure_pair_action(pair_action, B, device)

        Q, C = self.Q, self.C
        hold_index = Q * C
        is_hold = (pair_action == hold_index)

        idx = pair_action.clamp(max=hold_index - 1)
        i = torch.div(idx, C, rounding_mode='floor')
        j = (idx % C)

        batch_idx = torch.arange(B, device=device)
        job_tok  = J[batch_idx, i]
        core_tok = Cw[batch_idx, j]
        pair_h   = H[batch_idx, i, j]
        pair_ctx_d = self.PairCtxToD(pair_h)

        is_hold_f = is_hold.unsqueeze(-1)
        job_tok    = torch.where(is_hold_f, self.HoldJobEmb.expand_as(job_tok), job_tok)
        core_tok   = torch.where(is_hold_f, self.HoldCoreEmb.expand_as(core_tok), core_tok)
        pair_ctx_d = torch.where(is_hold_f, self.HoldPairCtx.expand_as(pair_ctx_d), pair_ctx_d)

        state_ctx = self._attn_pool(tokens, attn_mask)

        delay_in = torch.cat([state_ctx, job_tok, core_tok, pair_ctx_d], dim=-1)  # (B,4d)
        ab_raw = self.DelayHead(delay_in)  # (B,2)

        alpha = F.softplus(ab_raw[:, 0]) + eps
        beta  = F.softplus(ab_raw[:, 1]) + eps
        return alpha, beta, is_hold

    def delay_dist_with_action(self, x, pair_action, attn_mask=None):
        alpha, beta, is_hold = self.delay_params_with_action(x, pair_action, attn_mask)
        dist = torch.distributions.Beta(alpha, beta)
        return dist, is_hold

    def sample_delay_int(self, x, pair_action, max_delay, attn_mask=None, deterministic=False):
        """
        max_delay: scalar or (B,) giving allowed delay upper bound.
        Returns:
          delay_int: (B,) long in [0..max_delay]
          u:        (B,) float in (0,1)
          logp_u:   (B,) float log prob of u (0 for HOLD)
          entropy:  (B,) float entropy (0 for HOLD)
        """
        x = self._ensure_state(x)
        B = x.size(0)
        device = x.device

        pair_action = self._ensure_pair_action(pair_action, B, device)
        max_delay = self._ensure_max_delay(max_delay, B, device).clamp(min=0)

        dist, is_hold = self.delay_dist_with_action(x, pair_action, attn_mask)

        u = dist.mean if deterministic else dist.sample()
        logp_u = dist.log_prob(u)
        entropy = dist.entropy()

        delay_int = torch.round(u * max_delay.float()).long()

        delay_int = torch.where(is_hold, torch.zeros_like(delay_int), delay_int)
        logp_u    = torch.where(is_hold, torch.zeros_like(logp_u),    logp_u)
        entropy   = torch.where(is_hold, torch.zeros_like(entropy),   entropy)

        return delay_int, u, logp_u, entropy

    def get_outputs(self, x, pair_action=None, attn_mask=None):
        pair_logits = self.forward(x)
        delay_params = None
        if pair_action is not None:
            alpha, beta, is_hold = self.delay_params_with_action(x, pair_action, attn_mask)
            delay_params = (alpha, beta, is_hold)
        return pair_logits, delay_params


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
    def __init__(self, batch_size=681, inputNum_size=[9, 9, 1, 9], featureNum_size=[24, 8, 3, 5], delay_bins = 1000,
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
        self.delay_bins = delay_bins

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
        self.log_probs1 = []
        self.log_probs2 = []
        self.rewards_seq = []
        self.actions1 = []
        self.actions2 = []
        self.values = []
        self.masks1 = []
        self.masks2 = []
        self.job_inputs = []
        self.buffer = Buffer()

        self.ppo_update_time = 8
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.entropy_coefficient = 0

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=0.0001,eps=1e-6)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=0.0005,eps=1e-6)

    def act_assign(
            self,
            state: torch.Tensor,  # (B, rows, feat)
            pair_mask: torch.Tensor,  # (B, Q*C+1) bool
            attn_mask: torch.Tensor | None = None,  # (B, Q+M+C+G) bool
            device=None,
            sample: bool = True,
    ):
        """
        One rollout step (PAIR ONLY):
          1) pair logits -> sample/choose pair_action with mask
          2) critic value
        Returns:
          pair_action, pair_logprob, value, pair_logits
        """
        if device is not None:
            state = state.to(device)
            pair_mask = pair_mask.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

        # 1) Pair
        pair_logits = self.actor_net(state)  # (B, Q*C+1)
        dist_pair = CategoricalMasked(logits=pair_logits, masks=pair_mask, device=device)

        if sample:
            pair_action = dist_pair.sample()  # (B,)
        else:
            masked = pair_logits.masked_fill(~pair_mask, float("-inf"))
            pair_action = masked.argmax(dim=-1)

        pair_logprob = dist_pair.log_prob(pair_action)  # (B,)

        # 2) Critic
        value = self.critic_net(state).squeeze(-1)  # (B,)

        return (
            pair_action,
            pair_logprob,
            value,
            pair_logits,
        )

    import torch
    from torch.distributions import Beta

    def act_delay(
            self,
            state: torch.Tensor,
            pair_action: torch.Tensor,
            max_delay,
            attn_mask: torch.Tensor | None = None,
            device=None,
            sample: bool = True,
    ):
        if device is not None:
            state = state.to(device)
            pair_action = pair_action.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

        B = state.size(0)

        # ---- robust max_delay -> tensor (B,) ----
        if torch.is_tensor(max_delay):
            md = max_delay.to(state.device).long()
        else:
            md = torch.as_tensor(max_delay, device=state.device).long()

        if md.ndim == 0:
            md = md.view(1)
        if md.numel() == 1 and B > 1:
            md = md.expand(B)
        elif md.numel() != B:
            raise ValueError(f"max_delay has shape {tuple(md.shape)} but expected 1 or {B} elements")

        max_delay = md.clamp(min=0)

        # 1) Beta params
        alpha, beta, is_hold = self.actor_net.delay_params_with_action(state, pair_action, attn_mask)
        dist = Beta(alpha, beta)

        # 2) sample/mean u
        u = dist.sample() if sample else dist.mean

        delay_logprob = dist.log_prob(u)
        entropy = dist.entropy()

        # 3) integer delay
        delay_action = torch.round(u * max_delay.float()).long()

        # HOLD handling
        delay_action = torch.where(is_hold, torch.zeros_like(delay_action), delay_action)
        delay_logprob = torch.where(is_hold, torch.zeros_like(delay_logprob), delay_logprob)
        entropy = torch.where(is_hold, torch.zeros_like(entropy), entropy)

        delay_info = {
            "alpha": alpha,
            "beta": beta,
            "u": u,
            "entropy": entropy,
            "max_delay": max_delay,
            "is_hold": is_hold,
        }

        return delay_action, delay_logprob, delay_info

    def act_assign_delay(
            self,
            state: torch.Tensor,
            pair_actions: torch.Tensor,  # (B,)
            delay_actions: torch.Tensor,  # (B,) integer delay that env executed
            pair_mask: torch.Tensor,  # (B, Q*C+1) bool
            max_delay: int | torch.Tensor,  # scalar or (B,) upper bound used to map u->delay_int
            attn_mask: torch.Tensor | None = None,
            device=None,
            eps: float = 1e-6,
    ):
        """
        Evaluate logprobs/entropy for PPO given already-taken actions.

        Returns:
          pair_logprob, delay_logprob, pair_entropy, delay_entropy
        """
        if device is not None:
            state = state.to(device)
            pair_actions = pair_actions.to(device)
            delay_actions = delay_actions.to(device)
            pair_mask = pair_mask.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

        # ----- pair -----
        pair_logits = self.actor_net(state)  # (B, Q*C+1)
        dist_pair = CategoricalMasked(logits=pair_logits, masks=pair_mask, device=device)
        pair_logprob = dist_pair.log_prob(pair_actions)  # (B,)
        pair_entropy = dist_pair.entropy()  # (B,)

        # ----- make max_delay tensor (B,) -----
        if not torch.is_tensor(max_delay):
            max_delay = torch.tensor([max_delay], dtype=torch.long, device=state.device)
            if state.size(0) > 1:
                max_delay = max_delay.expand(state.size(0))
        else:
            max_delay = max_delay.to(state.device).long()
            if max_delay.ndim == 0:
                max_delay = max_delay.expand(state.size(0))

        max_delay = max_delay.clamp(min=0)

        # ----- delay: Beta(alpha,beta) conditioned on pair_actions -----
        alpha, beta, is_hold = self.actor_net.delay_params_with_action(state, pair_actions, attn_mask)
        dist_delay = Beta(alpha, beta)

        # Reconstruct u from integer delay_action and max_delay
        # If max_delay==0 -> define u=0.5 (arbitrary), but HOLD will be handled below
        denom = torch.where(max_delay > 0, max_delay.float(), torch.ones_like(max_delay).float())
        u = (delay_actions.float() / denom).clamp(eps, 1.0 - eps)

        delay_logprob = dist_delay.log_prob(u)  # (B,)
        delay_entropy = dist_delay.entropy()  # (B,)

        # HOLD handling: neutralize delay terms (same as during acting)
        delay_logprob = torch.where(is_hold, torch.zeros_like(delay_logprob), delay_logprob)
        delay_entropy = torch.where(is_hold, torch.zeros_like(delay_entropy), delay_entropy)

        return (
            pair_logprob,
            delay_logprob,
            pair_entropy,
            delay_entropy
        )

    def normalize(self, advantages):
        nor_advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-9)
        return nor_advantages

    # def remember(self, state, value, log_prob1, log_prob2, action1, action2, reward, mask1, mask2, device):
    #     self.rewards_seq.append(reward)
    #     self.states.append(state.to("cpu"))
    #     self.log_probs1.append(log_prob1.to("cpu"))
    #     self.log_probs2.append(log_prob2.to("cpu"))
    #     self.values.append(value.to("cpu"))
    #     self.actions1.append(action1.to("cpu"))
    #     self.actions2.append(action2.to("cpu"))
    #     self.masks1.append(mask1.to("cpu"))
    #     self.masks2.append(mask2.to("cpu"))
    #     # self.job_inputs.append(job_input.to("cpu"))
    def remember(self, state, value, log_prob1, log_prob2, action1, action2,
                 reward, mask1, mask2, device):
        self.rewards_seq.append(reward)

        self.states.append(state.detach().to("cpu"))
        self.log_probs1.append(log_prob1.detach().to("cpu"))
        self.log_probs2.append(log_prob2.detach().to("cpu"))
        self.values.append(value.detach().to("cpu"))
        self.actions1.append(action1.detach().to("cpu"))
        self.actions2.append(action2.detach().to("cpu"))

        # mask1 should be a torch tensor; if it's numpy, convert
        if not torch.is_tensor(mask1):
            mask1 = torch.as_tensor(mask1, dtype=torch.bool, device=state.device)
        self.masks1.append(mask1.detach().to("cpu"))

        # mask2 is actually max_delay now; make it a Long tensor
        if torch.is_tensor(mask2):
            md = mask2.detach()
        else:
            # numpy scalar/array or python int -> tensor
            md = torch.as_tensor(mask2, dtype=torch.long, device=state.device)

        # ensure it's at least 1D or scalar is fine; just store on cpu
        self.masks2.append(md.to("cpu"))

    def clear_memory(self):
        self.rewards_seq = []
        self.states = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.values = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []

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

        self.buffer.store_buffer(self.states, self.masks1, self.masks2, self.actions1, self.actions2, self.log_probs1,
                                 self.log_probs2,
                                 returns, advantages, len(self.states))

    def compute_value_loss(self, states, returns):
        state_values = self.critic_net(states)
        state_values = torch.squeeze(state_values, dim=1)

        # Calculate value loss using F.mse_loss
        value_loss = F.mse_loss(state_values, returns)
        return value_loss

    def compute_actor_loss(
            self,
            states,
            pair_mask,
            pair_actions,
            delay_actions,
            max_delay,  # NEW: needed for delay logprob reconstruction
            advantages,
            old_log_probs1,
            old_log_probs2
    ):
        # NEW: evaluate current policy logprobs under Beta delay head
        log_probs1, log_probs2, entropy1, entropy2 = self.act_assign_delay(
            states,
            pair_actions,
            delay_actions,
            pair_mask,
            max_delay,
            device=self.device
        )

        # PPO ratio for joint action (pair + delay)
        log_old = old_log_probs1 + old_log_probs2
        log_new = log_probs1 + log_probs2
        logratio = log_new - log_old
        ratio = torch.exp(logratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # entropy bonus
        entropy = (entropy1 + entropy2) / 2.0
        entropy_loss = torch.mean(entropy)

        total_loss = policy_loss - self.entropy_coefficient * entropy_loss
        return total_loss, policy_loss, entropy_loss

    def train(self):
        states = torch.cat(self.buffer.states, dim=0)
        masks1 = torch.cat(self.buffer.masks1, dim=0)
        masks2 = torch.cat(self.buffer.masks2, dim=0)
        actions1 = torch.cat(self.buffer.actions1, dim=0)
        log_probs1 = torch.cat(self.buffer.log_probs1, dim=0)
        actions2 = torch.cat(self.buffer.actions2, dim=0)
        log_probs2 = torch.cat(self.buffer.log_probs2, dim=0)
        returns = torch.tensor(self.buffer.Returns)
        advantages = torch.tensor(self.buffer.advantages)
        advantages = self.normalize(advantages)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), self.batch_size, False):
                index_tensor = torch.tensor(index)
                sampled_states = torch.index_select(states, dim=0, index=index_tensor).to(self.device)
                sampled_masks1 = torch.index_select(masks1, dim=0, index=index_tensor).to(self.device)
                sampled_masks2 = torch.index_select(masks2, dim=0, index=index_tensor).to(self.device)
                sampled_actions1 = torch.index_select(actions1, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs1 = torch.index_select(log_probs1, dim=0, index=index_tensor).to(self.device)
                sampled_actions2 = torch.index_select(actions2, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs2 = torch.index_select(log_probs2, dim=0, index=index_tensor).to(self.device)
                sampled_returns = torch.index_select(returns, dim=0, index=index_tensor).to(self.device)
                sampled_advantages = torch.index_select(advantages, dim=0, index=index_tensor).to(self.device)

                self.actor_optimizer.zero_grad()
                action_loss, value_loss, entropy_loss = self.compute_actor_loss(sampled_states,
                                                                                sampled_masks1,
                                                                                sampled_masks2,
                                                                                sampled_actions1,
                                                                                sampled_actions2,
                                                                                sampled_advantages,
                                                                                sampled_log_probs1,
                                                                                sampled_log_probs2)
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_net_optimizer.zero_grad()
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)
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

    def eval_action(self,
                    o,
                    mask1,
                    mask2,
                    attn_mask = None,
                    device = None,
                    greedy: bool = True,
                    ):
        with torch.no_grad():
            # o = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
            o = o.reshape(1, 9 + 9 + 1 + 9, 24)
            state = torch.FloatTensor(o).to(self.device)
            # mask = np.array(mask).reshape(1, MAX_QUEUE_SIZE + run_win + green_win)
            mask1 = torch.FloatTensor(mask1).to(self.device)
            mask2 = torch.FloatTensor(mask2).to(self.device)
            pair_logits = self.actor_net(state)  # (1, Q*C+1), logits only

            if greedy:
                masked = pair_logits.masked_fill(~mask1, float("-inf"))
                pair_action = masked.argmax(dim=-1)  # (1,)
            else:
                dist_pair = CategoricalMasked(logits=pair_logits, masks=mask1, device=device)
                pair_action = dist_pair.sample()  # (1,)

                # ---- delay action (conditioned on chosen pair) ----
            delay_logits = self.actor_net.delay_logits_with_action(state, pair_action, attn_mask)  # (1, K)

            if greedy:
                masked = delay_logits.masked_fill(~mask2, float("-inf"))
                delay_action = masked.argmax(dim=-1)  # (1,)
            else:
                dist_delay = CategoricalMasked(logits=delay_logits, masks=mask2, device=device)
                delay_action = dist_delay.sample()  # (1,)

                    # return python ints
        return int(pair_action.item()), int(delay_action.item())
                                 ## mask should to be cheacked !






def train(env):
    # seed = 0
    epochs = 300
    traj_num = 100
    # env = SchedulerEnv(scheduler, action_decoder)
    # env.seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # inputNum_size   = [waiting_queue, running_queue, green, core_queue]
    inputNum_size     = [9, 9, 1, 9]
    # featureNum_size = [JOB_FEATURES, RUN_FEATURES, GREEN_FEATURE, CORE_FEATURES]
    featureNum_size   = [24, 8, 3, 5]
    ep_len_int = 0
    ppo = PPO(batch_size=ep_len_int, inputNum_size=inputNum_size,
              featureNum_size=featureNum_size, device=device)
    for epoch in range(epochs):
        print(epoch)
        o, _ = env.reset()
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
                o = o.reshape(1, 9 + 9 + 1 + 9, 24)
                state = torch.FloatTensor(o).to(device)
                mask1 = env.get_valid_assign_mask() ## for now just assign_task ! next we need delay as well !
                mask1 = torch.tensor(np.stack(mask1), dtype=torch.bool, device=device)
                # print(mask1)
                # mask2 = torch.tensor(np.stack(mask2), dtype=torch.bool, device=device)
                if mask1.ndim == 1:
                    mask1 = mask1.unsqueeze(0)  # (1, 82)
                # if mask2.ndim == 1:
                #     mask2 = mask2.unsqueeze(0)  # (1, 82)
                # print(mask.shape, "org")
                # pack that 3 up code into 1 code next time !
                # if mask.any() :
                #     print(mask)
                pair_action,  pair_logprob,   value, pair_logits, = ppo.act_assign(state, mask1)
                #=================================================
                mask_delay = env.get_valid_delay_mask(pair_action)
                # print(f'max delay : {mask_delay}')
                # =================================================
                delay_action, delay_logprob, _ =ppo.act_delay(state, pair_action, mask_delay)
            ppo.remember(state, value,pair_logprob, delay_logprob, pair_action, delay_action, epoch_reward, mask1, mask_delay, device)

            # print(print(f'agent : {pair_action.item()}'))
            # if pair_action != 180 :
            delay_action = int(delay_action)
            # if delay_action != 0:
            #     print(delay_action)
            o, r, d, truncated, infos = env.step((pair_action.item(),100))
            ep_ret += r
            ep_len += 1
            if ep_ret != 0 :
                print(ep_ret, ep_len, 'time:',env.scheduler.t, "nice_task:",env.scheduler.finish_on_time)
            # show_ret += r2
            # sjf += sjf_t
            # f1 += f1_t
            total_r, rt, wt, st = env.scheduler._compute_reward()

            run_time                 += rt
            waiting                   += wt
            slack                    += st
            epoch_reward             += total_r

            if d:
                t += 1
                ppo.storeIntoBuffter(r)
                ppo.clear_memory()
                o, _ = env.reset()
                r = 0
                d = False
                ep_ret = 0
                ep_len = 0
                print(f'{t} : traj_num')
                if t >= traj_num:
                    break

        ppo.train()
        with open('MaskablePPO_' + "16landa" + '.csv', mode='a',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow([float(epoch_reward / traj_num),float(run_time / traj_num),float(waiting / traj_num),float(slack / traj_num)])
        ppo.buffer.clear_buffer()

    ppo.save_using_model_name('Each_16' + '/MaskablePPO')
