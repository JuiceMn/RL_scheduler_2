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
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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
        self.actions1 = []        # pair_action (int)
        self.actions2 = []        # u (float in (0,1))  <-- Fix A
        self.masks1 = []          # pair_mask
        self.masks2 = []          # max_delay (long)
        self.log_probs1 = []      # log pi(pair_action)
        self.log_probs2 = []      # log pi(u)
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

    def store_buffer(self,
                     state,
                     mask1,
                     max_delay,
                     action1,
                     u,                 # <-- Fix A: store u instead of delay_int
                     log_prob1,
                     log_prob2,
                     Return,
                     advantage,
                     nums):
        """
        Fix A buffer:
          - action1: pair_action
          - action2: u (Beta sample)
          - masks2: max_delay (still stored)
        """
        self.buffer_num += nums

        self.states.extend(state)
        self.masks1.extend(mask1)
        self.masks2.extend(max_delay)
        self.actions1.extend(action1)
        self.actions2.extend(u)            # <-- Fix A
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
            nn.Linear(4 * d_model + 1, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

        # ---- put this at the end of ActorNet.__init__, right after self.DelayHead is created ----
        def softplus_inv(y: torch.Tensor) -> torch.Tensor:
            # inverse of softplus for y>0
            return torch.log(torch.expm1(torch.clamp(y, min=1e-8)))

        u0 = 0.000001  # initial mean of u (small delay fraction)
        kappa = 60.0  # concentration (bigger => less random)

        alpha0 = torch.tensor(u0 * kappa)
        beta0 = torch.tensor((1.0 - u0) * kappa)

        last = self.DelayHead[-1]  # the final nn.Linear(256 -> 2)

        # set bias so softplus(bias) ~= (alpha0, beta0)
        with torch.no_grad():
            b = torch.stack([softplus_inv(alpha0), softplus_inv(beta0)]).to(last.bias.device)
            last.bias.copy_(b)

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
    def delay_params_with_action(self, x, pair_action, max_delay=None, attn_mask=None, eps=1e-4):
        """
        Returns (alpha, beta, is_hold).
        alpha,beta: (B,) positive
        """

        J, _, _, Cw, _, H, tokens = self._encode_rows(x)
        B = tokens.size(0)
        device = tokens.device
        if max_delay is None:
            md = torch.zeros(B, dtype=torch.long, device=device)
        else:
            if torch.is_tensor(max_delay):
                md = max_delay.to(device).long().view(-1)
            else:
                md = torch.as_tensor(max_delay, device=device).long().view(-1)

            if md.numel() == 1 and B > 1:
                md = md.expand(B)
            elif md.numel() != B:
                raise ValueError(f"max_delay shape {md.shape} but expected 1 or {B}")

        md = md.clamp(min=0)
        md_norm = torch.log1p(md.float()).unsqueeze(-1)  # (B,1)
        # md_norm = md_norm / 10.0
        no_delay = (md == 0)
        md_norm = torch.where(no_delay.unsqueeze(-1), torch.zeros_like(md_norm), md_norm)

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


        md_norm = torch.where(is_hold.unsqueeze(-1), torch.zeros_like(md_norm), md_norm)


        is_hold_f = is_hold.unsqueeze(-1)
        job_tok    = torch.where(is_hold_f, self.HoldJobEmb.expand_as(job_tok), job_tok)
        core_tok   = torch.where(is_hold_f, self.HoldCoreEmb.expand_as(core_tok), core_tok)
        pair_ctx_d = torch.where(is_hold_f, self.HoldPairCtx.expand_as(pair_ctx_d), pair_ctx_d)

        state_ctx = self._attn_pool(tokens, attn_mask)

        delay_in = torch.cat([state_ctx, job_tok, core_tok, pair_ctx_d, md_norm], dim=-1)  # (B,4d+1)
        ab_raw = self.DelayHead(delay_in)

        alpha = F.softplus(ab_raw[:, 0]) + eps
        beta  = F.softplus(ab_raw[:, 1]) + eps
        return alpha, beta, is_hold

    def delay_dist_with_action(self, x, pair_action, attn_mask=None):
        alpha, beta, is_hold = self.delay_params_with_action(x, pair_action, max_delay=None, attn_mask=attn_mask)
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
            alpha, beta, is_hold = self.delay_params_with_action(x, pair_action, max_delay=None, attn_mask=attn_mask)
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

        self.ppo_update_time = 5
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        # entropy weights (separate!)
        self.ent_coef_pair = 1e-3  # categorical entropy (pair head)
        self.lr_shared = 3e-5  # shared trunk (encoders + shared projections)
        self.lr_pair = 1e-4  # pair head (categorical)
        self.lr_delay = 1e-5  # delay head (Beta) usually smaller
        self.ent_coef_delay = 1e-4  # beta entropy (delay head) usually much smaller

        self.lr_shared0 = self.lr_shared
        self.lr_pair0 = self.lr_pair
        self.lr_delay0 = self.lr_delay
        self.target_kl = 0.02  # common: 0.01 to 0.05
        # (optional) anneal critic too
        self.lr_critic0 = 5e-4  # whatever you use

        self.device = device
        self.actor_net = ActorNet(
            self.num_inputs1, self.num_inputs2, self.num_inputs3, self.num_inputs4, self.featureNum1,
            self.featureNum2, self.featureNum3, self.featureNum4).to(self.device)
        # -----------------------------
        # Actor optimizer with param groups (different LRs)
        # -----------------------------

        # 1) pair-head-only params (categorical logits for (task,core) + HOLD)
        pair_only = (
                list(self.actor_net.PairScore.parameters()) +
                list(self.actor_net.HoldHead.parameters())
        )

        # 2) delay-head-only params (Beta(alpha,beta) head + its pooling machinery)
        delay_only = (
                list(self.actor_net.DelayHead.parameters()) +
                list(self.actor_net.TypeEmb.parameters()) +
                list(self.actor_net.AttnWk.parameters()) +
                list(self.actor_net.PairCtxToD.parameters()) +
                [
                    self.actor_net.attn_q,
                    self.actor_net.HoldJobEmb,
                    self.actor_net.HoldCoreEmb,
                    self.actor_net.HoldPairCtx,
                ]
        )

        # 3) shared trunk params = everything else (encoders + shared projections/fusion)
        pair_ids = {id(p) for p in pair_only}
        delay_ids = {id(p) for p in delay_only}

        shared = [p for p in self.actor_net.parameters()
                  if id(p) not in pair_ids and id(p) not in delay_ids]

        # Optional sanity checks (run once)
        assert pair_ids.isdisjoint(delay_ids), "pair_only and delay_only overlap!"
        assert all(id(p) not in pair_ids and id(p) not in delay_ids for p in shared), "shared overlaps!"

        # Create ONE optimizer with 3 param groups (different LR per group)
        self.actor_optimizer = optim.Adam(
            [
                {"params": shared, "lr": self.lr_shared},
                {"params": pair_only, "lr": self.lr_pair},
                {"params": delay_only, "lr": self.lr_delay},
            ],
            eps=1e-6
        )
        self.critic_net = CriticNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3, self.featureNum4, self.num_inputs4).to(self.device)


        # self.actor_optimizer = optim.Adam(
        #     self.actor_net.parameters(), lr=0.0001,eps=1e-6)
        # print("shared:", sum(p.numel() for p in shared))
        # print("pair_only:", sum(p.numel() for p in pair_only))
        # print("delay_only:", sum(p.numel() for p in delay_only))
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=0.0005,eps=1e-6)

    def set_lrs(self, progress_remaining: float):
        """
        progress_remaining: 1.0 at start -> 0.0 at end
        """
        pr = float(np.clip(progress_remaining, 0.0, 1.0))

        # actor optimizer param_groups: [shared, pair_only, delay_only]
        self.actor_optimizer.param_groups[0]["lr"] = self.lr_shared0 * pr
        self.actor_optimizer.param_groups[1]["lr"] = self.lr_pair0 * pr
        self.actor_optimizer.param_groups[2]["lr"] = self.lr_delay0 * pr

        # (optional) critic LR anneal
        for g in self.critic_net_optimizer.param_groups:
            g["lr"] = self.lr_critic0 * pr

    # def beta_round_delay_logprob(self, dist, delay_int, max_delay, eps=1e-12):
    #     """
    #     Log prob of executed integer delay when integer is produced by rounding:
    #         delay_int = round(u * max_delay),   u ~ Beta(alpha,beta)
    #
    #     Uses interval mass via Beta CDF:
    #       P(k) = F(upper_k) - F(lower_k)
    #
    #     NOTE: torch.distributions.Beta has no .cdf() in many torch versions,
    #           so we use torch.special.betainc(a,b,x) instead.
    #     """
    #     device = delay_int.device
    #     md = max_delay.to(device).long()
    #     k = delay_int.to(device).long()
    #
    #     md = md.clamp(min=0)
    #     k = k.clamp(min=0)
    #
    #     # ensure executed delay is within [0..md]
    #     k = torch.minimum(k, md)
    #
    #     # md == 0 => only k==0 possible with prob 1
    #     is_zero = (md == 0)
    #
    #     md_f = md.float().clamp(min=1.0)
    #
    #     lower = (k.float() - 0.5) / md_f
    #     upper = (k.float() + 0.5) / md_f
    #
    #     # rounding edge bins
    #     lower = torch.where(k == 0, torch.zeros_like(lower), lower)
    #     upper = torch.where(k == md, torch.ones_like(upper), upper)
    #
    #     lower = lower.clamp(0.0, 1.0)
    #     upper = upper.clamp(0.0, 1.0)
    #
    #     # ---- Beta CDF via regularized incomplete beta ----
    #     if not hasattr(torch.special, "betainc"):
    #         raise RuntimeError(
    #             "torch.special.betainc not found in your PyTorch build. "
    #             "You cannot use Fix B (CDF interval) unless you upgrade PyTorch, "
    #             "or switch to Fix A (store u and use dist.log_prob(u))."
    #         )
    #
    #     a = dist.concentration1
    #     b = dist.concentration0
    #
    #     # (optional but recommended) do betainc in float64 for stability
    #     lower64 = lower.to(torch.float64)
    #     upper64 = upper.to(torch.float64)
    #     a64 = a.to(torch.float64)
    #     b64 = b.to(torch.float64)
    #
    #     F_upper = torch.special.betainc(a64, b64, upper64)
    #     F_lower = torch.special.betainc(a64, b64, lower64)
    #
    #     p_nz = (F_upper - F_lower).to(torch.float32)
    #     p_nz = p_nz.clamp(min=eps)
    #
    #     p = torch.where(is_zero, (k == 0).float(), p_nz)
    #     return torch.log(p.clamp(min=eps))

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

    from torch.distributions import Beta

    def act_delay(
            self,
            state: torch.Tensor,
            pair_action: torch.Tensor,
            max_delay,
            attn_mask: torch.Tensor | None = None,
            device=None,
            sample: bool = True,
            eps: float = 1e-6,
    ):
        """
        Fix A rollout:
          - u ~ Beta(alpha,beta)
          - execute delay_int = round(u * max_delay)
          - store logprob(u) for PPO (NOT probability of the integer delay)
          - HOLD and max_delay==0 => delay_int=0, u_logprob=0, entropy=0
        Returns:
          delay_int (B,) long
          u (B,) float
          u_logprob (B,) float
          delay_info dict
        """
        if device is not None:
            state = state.to(device)
            pair_action = pair_action.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

        B = state.size(0)

        # ---- robust max_delay -> (B,) long ----
        if torch.is_tensor(max_delay):
            md = max_delay.to(state.device).long().view(-1)
        else:
            md = torch.as_tensor(max_delay, device=state.device).long().view(-1)

        if md.numel() == 1 and B > 1:
            md = md.expand(B)
        elif md.numel() != B:
            raise ValueError(f"max_delay has shape {tuple(md.shape)} but expected 1 or {B} elements")

        md = md.clamp(min=0)
        no_delay = (md == 0)

        # ---- Beta params (pass max_delay because delay head uses it now) ----
        alpha, beta, is_hold = self.actor_net.delay_params_with_action(
            state, pair_action, max_delay=md, attn_mask=attn_mask
        )
        dist = Beta(alpha, beta)

        # sample/mean u
        u = dist.sample() if sample else dist.mean
        u = u.clamp(eps, 1.0 - eps)  # avoid log_prob inf

        u_logprob = dist.log_prob(u)
        entropy = dist.entropy()

        # integer delay to execute
        delay_int = torch.round(u * md.float()).long()
        delay_int = torch.clamp(delay_int, min=0)
        delay_int = torch.minimum(delay_int, md)

        # HOLD or no_delay => force delay=0 and neutralize delay terms
        neutral = (is_hold | no_delay)
        delay_int = torch.where(neutral, torch.zeros_like(delay_int), delay_int)
        u_logprob = torch.where(neutral, torch.zeros_like(u_logprob), u_logprob)
        entropy = torch.where(neutral, torch.zeros_like(entropy), entropy)

        delay_info = {
            "alpha": alpha,
            "beta": beta,
            "u": u,
            "entropy": entropy,
            "max_delay": md,
            "is_hold": is_hold,
            "no_delay": no_delay,
        }
        return delay_int, u, u_logprob, delay_info

    def act_assign_delay(
            self,
            state: torch.Tensor,
            pair_actions: torch.Tensor,  # (B,)
            u_actions: torch.Tensor,  # (B,) stored u from rollout  <-- Fix A
            pair_mask: torch.Tensor,  # (B, Q*C+1)
            max_delay,  # scalar or (B,)
            attn_mask: torch.Tensor | None = None,
            device=None,
            eps: float = 1e-6,
    ):
        """
        Fix A PPO eval:
          - pair: masked categorical logprob + entropy
          - delay: Beta logprob(u_actions) + entropy
          - neutralize delay terms for HOLD and max_delay==0
        Returns:
          pair_logprob, u_logprob, pair_entropy, delay_entropy
        """
        if device is not None:
            state = state.to(device)
            pair_actions = pair_actions.to(device)
            u_actions = u_actions.to(device)
            pair_mask = pair_mask.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

        B = state.size(0)

        # ----- pair head -----
        pair_logits = self.actor_net(state)
        dist_pair = CategoricalMasked(logits=pair_logits, masks=pair_mask, device=device)
        pair_actions = pair_actions.long().view(-1)

        pair_logprob = dist_pair.log_prob(pair_actions)
        pair_entropy = dist_pair.entropy()

        # ----- max_delay -> (B,) -----
        if torch.is_tensor(max_delay):
            md = max_delay.to(state.device).long().view(-1)
        else:
            md = torch.as_tensor(max_delay, device=state.device).long().view(-1)

        if md.numel() == 1 and B > 1:
            md = md.expand(B)
        elif md.numel() != B:
            raise ValueError(f"max_delay has shape {tuple(md.shape)} but expected 1 or {B} elements")

        md = md.clamp(min=0)
        no_delay = (md == 0)

        # ----- delay head -----
        alpha, beta, is_hold = self.actor_net.delay_params_with_action(
            state, pair_actions, max_delay=md, attn_mask=attn_mask
        )
        dist_delay = Beta(alpha, beta)

        u_actions = u_actions.float().view(-1).clamp(eps, 1.0 - eps)
        u_logprob = dist_delay.log_prob(u_actions)
        delay_entropy = dist_delay.entropy()

        # Neutralize delay terms when irrelevant
        neutral = (is_hold | no_delay)
        u_logprob = torch.where(neutral, torch.zeros_like(u_logprob), u_logprob)
        delay_entropy = torch.where(neutral, torch.zeros_like(delay_entropy), delay_entropy)

        return pair_logprob, u_logprob, pair_entropy, delay_entropy

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
    import torch

    def remember(self, state, value, log_prob1, log_prob2,
                 action1, u, reward, mask1, max_delay, device):
        """
        Fix A:
          action1 = pair_action (int)
          u       = sampled Beta fraction in (0,1)  (float)
          log_prob2 = log_prob(u)
          max_delay stored for reference/execution scale
        """
        self.rewards_seq.append(reward)

        self.states.append(state.detach().to("cpu"))
        self.log_probs1.append(log_prob1.detach().to("cpu"))
        self.log_probs2.append(log_prob2.detach().to("cpu"))  # log_prob(u)
        self.values.append(value.detach().to("cpu"))

        # ---- pair action (store as (1,) long) ----
        a1 = action1 if torch.is_tensor(action1) else torch.as_tensor(action1)
        self.actions1.append(a1.detach().to("cpu").long().view(1))

        # ---- u action (store as (1,) float) ----
        u_t = u if torch.is_tensor(u) else torch.as_tensor(u)
        self.actions2.append(u_t.detach().to("cpu").float().view(1))

        # ---- pair mask (bool) ----
        if not torch.is_tensor(mask1):
            mask1 = torch.as_tensor(mask1, dtype=torch.bool, device=state.device)
        self.masks1.append(mask1.detach().to("cpu"))

        # ---- max_delay (store as (1,) long) ----
        if torch.is_tensor(max_delay):
            md = max_delay.detach().to("cpu").long().view(-1)
        else:
            md = torch.as_tensor(max_delay, dtype=torch.long).to("cpu").view(-1)
        if md.numel() != 1:
            md = md[:1]
        self.masks2.append(md.view(1))

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

    import torch
    import torch.nn.functional as F
    from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

    def compute_value_loss(self, states, returns):
        returns = returns.to(states.device).float()
        state_values = self.critic_net(states).squeeze(-1)  # (B,)
        return F.mse_loss(state_values, returns)

    def compute_actor_loss(
            self,
            states,
            pair_mask,
            max_delay,
            pair_actions,
            u_actions,  # Fix A: u in (0,1)
            advantages,
            old_logp_pair,
            old_logp_u
    ):
        pair_actions = pair_actions.long().view(-1)
        u_actions = u_actions.float().view(-1)
        max_delay = max_delay.long().view(-1)

        advantages = advantages.view(-1)
        old_logp_pair = old_logp_pair.view(-1)
        old_logp_u = old_logp_u.view(-1)

        # current logprobs + entropy
        logp_pair, logp_u, ent_pair, ent_u = self.act_assign_delay(
            states,
            pair_actions=pair_actions,
            u_actions=u_actions,
            pair_mask=pair_mask,
            max_delay=max_delay,
            device=self.device
        )

        # ----------------------------
        # Metro-style: separate PPO ratios
        # ----------------------------
        ratio_pair = torch.exp(logp_pair - old_logp_pair)
        ratio_u = torch.exp(logp_u - old_logp_u)

        # clipped surrogate per head (same advantage)
        surr1_pair = ratio_pair * advantages
        surr2_pair = torch.clamp(ratio_pair, 1 - self.clip_param, 1 + self.clip_param) * advantages
        loss_pair = -torch.mean(torch.min(surr1_pair, surr2_pair))

        surr1_u = ratio_u * advantages
        surr2_u = torch.clamp(ratio_u, 1 - self.clip_param, 1 + self.clip_param) * advantages
        loss_u = -torch.mean(torch.min(surr1_u, surr2_u))

        # sum the two clipped objectives
        w_pair = 1.0
        w_u = 0.25  # test this shit !
        policy_loss = w_pair * loss_pair + w_u * loss_u
        # policy_loss = loss_pair + loss_u

        # entropy bonus separately weighted (you already do this)
        entropy_bonus = self.ent_coef_pair * ent_pair.mean() + self.ent_coef_delay * ent_u.mean()

        total_loss = policy_loss - entropy_bonus
        return total_loss, policy_loss, entropy_bonus

    def train(self):
        # ---- stack buffer ----
        states = torch.cat(self.buffer.states, dim=0)  # (N, rows, feat)
        masks1 = torch.cat(self.buffer.masks1, dim=0).bool()  # (N, Q*C+1)
        maxdelay = torch.cat(self.buffer.masks2, dim=0).view(-1).long()  # (N,)

        actions1 = torch.cat(self.buffer.actions1, dim=0).view(-1).long()  # (N,)
        u_actions = torch.cat(self.buffer.actions2, dim=0).view(-1).float()  # (N,)  <-- Fix A

        logp1 = torch.cat(self.buffer.log_probs1, dim=0).view(-1)  # (N,)
        logp2 = torch.cat(self.buffer.log_probs2, dim=0).view(-1)  # (N,)  log_prob(u)

        returns = torch.tensor(self.buffer.Returns, dtype=torch.float32)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = self.normalize(advantages)

        N = states.size(0)

        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(N)), self.batch_size, drop_last=False):
                idx = torch.tensor(index, dtype=torch.long)

                sampled_states = states.index_select(0, idx).to(self.device)
                sampled_masks1 = masks1.index_select(0, idx).to(self.device)
                sampled_maxdelay = maxdelay.index_select(0, idx).to(self.device)

                sampled_a1 = actions1.index_select(0, idx).to(self.device)
                sampled_u = u_actions.index_select(0, idx).to(self.device)  # <-- Fix A
                sampled_logp1 = logp1.index_select(0, idx).to(self.device)
                sampled_logp2 = logp2.index_select(0, idx).to(self.device)

                sampled_ret = returns.index_select(0, idx).to(self.device)
                sampled_adv = advantages.index_select(0, idx).to(self.device)

                # ---- Actor update ----
                self.actor_optimizer.zero_grad()

                total_loss, policy_loss, entropy_bonus = self.compute_actor_loss(
                    sampled_states,
                    sampled_masks1,
                    sampled_maxdelay,
                    sampled_a1,
                    sampled_u,  # <-- Fix A
                    sampled_adv,
                    sampled_logp1,
                    sampled_logp2
                )
                # early_stop = False
                # if approx_kl.item() > 1.5 * self.target_kl:
                #     early_stop = True
                #     break
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                # if early_stop:
                #     break

                # ---- Critic update ----
                self.critic_net_optimizer.zero_grad()

                value_loss = self.compute_value_loss(sampled_states, sampled_ret)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
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
                # 1) sample pair
                pair_action, pair_logprob, value, _ = ppo.act_assign(state, mask1)
                #=================================================
                # 2) get max_delay (upper bound) for this selected pair
                pair_a = int(pair_action.item())
                max_delay = env.get_valid_delay_mask(pair_a)
                # =================================================
                # 3) sample delay using Fix A
                delay_int, u, u_logprob, info = ppo.act_delay(state, pair_action, max_delay)

            # print(print(f'agent : {pair_action.item()}'))
            # if pair_action != 180 :
            # delay_action = int(delay_action)

            # print(delay_action)
            o, r, d, truncated, infos = env.step((pair_action.item(),int(delay_int)))
            ep_ret += r
            ep_len += 1
            if ep_ret != 0 :
                print('reward is :', ep_ret, ep_len,'mean delay:', env.scheduler.mean_delay, 'time:',env.scheduler.t, "nice_task:",env.scheduler.finish_on_time,
                        "all execution : ", sum(env.scheduler.runtime_reward_vector), "all energy", sum(env.scheduler.energy_reward_vector),
                      "total wipe:", env.scheduler.total_wipe,
                      "total miss :", env.scheduler.active_queue_miss_counter,
                      "all task :",   env.scheduler.task_name_finished  )
            # show_ret += r2
            # sjf += sjf_t
            # f1 += f1_t
            total_r, rt, wt, st = env.scheduler._compute_reward()

            run_time                 += rt
            waiting                   += wt
            slack                    += st
            epoch_reward             += total_r
            # 4) store into PPO buffer (Fix A stores u, not delay_int)
            ppo.remember(
                state,
                value,
                pair_logprob,
                u_logprob,  # log_prob2 = log_prob(u)
                pair_action,  # action1
                u,  # action2 = u (float)
                total_r,
                mask1,
                max_delay,  # masks2 = max_delay
                device
            )
            if d:
                t += 1
                ppo.storeIntoBuffter(r)
                ppo.clear_memory()
                o, _ = env.reset()
                r = 0
                d = False
                ep_len_int = ep_len
                ppo.batch_size = ep_len_int
                ep_ret = 0
                ep_len = 0
                print(f'{t} : traj_num')
                if t >= traj_num:
                    break
        progress_remaining = 1.0 - (epoch / max(1, epochs - 1))
        ppo.set_lrs(progress_remaining)
        ppo.train()
        with open('MaskablePPO_' + "16landa" + '.csv', mode='a',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerow([float(epoch_reward / traj_num),float(run_time / traj_num),float(waiting / traj_num),float(slack / traj_num)])
        ppo.buffer.clear_buffer()

    ppo.save_using_model_name('Each_16' + '/MaskablePPO')
