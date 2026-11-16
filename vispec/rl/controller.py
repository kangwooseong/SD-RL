"""RL controller for dynamic draft depth / verification size in ViSpec.

The controller is attached to the draft model as ``spec_layer.rl_controller``.
``Model.topK_genrate`` and ``SpecModel.specgenerate`` call the hooks below.
When no controller is attached, ViSpec behaves exactly as before.

Decisions per draft-and-verify cycle
  * depth policy: after each tree expansion, continue (1) or stop (0)
  * size policy : how many draft tokens are sent to the target for verification

The policies only change the shape of the draft tree. Acceptance is still decided by
the target model, so the output distribution of the target is untouched.

Reward (cfg.reward_mode)
  * "gain" (default):  r = gain_scale * (tokens - rate_ref * cycle_time)
        rate_ref = reference throughput in tokens/sec. It starts at the throughput of
        plain ViSpec and is refreshed once per PPO update, so r > 0 means "this cycle
        was faster than the reference". The sum of r over cycles is meaningful, so
        returns are accumulated across the cycles of one generation (gamma, GAE).
        At convergence this maximises total tokens / total time.
  * "throughput":      r = throughput_scale * tokens / cycle_time
        per-cycle ratio as in Learning to Draft; every cycle is its own episode.

cycle_time = draft time + verification time, either measured (CUDA synchronised) or
predicted by the fitted CostModel (time_source = "model").
"""

import atexit
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .cost_model import CostModel

MODES = ("fixed", "random", "policy")


@dataclass
class RLConfig:
    top_k: int = 8  # must equal the draft tree width (--top-k)
    hidden_size: int = 4096  # hidden size of the target LLM (dim of g_v)
    min_depth: int = 1  # expansions that always happen
    max_depth: int = 6  # hard cap on expansions
    # candidate values for --total-token (root included, like the ViSpec flag)
    size_choices: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60])
    size_topn: int = 64  # how many of the best tree scores the size policy sees
    hist_len: int = 4  # number of past cycles kept in the state
    vis_dim: int = 32  # g_v is projected to this many dims inside the policy
    use_visual: bool = True  # False = ablation without visual features
    pi_hidden: List[int] = field(default_factory=lambda: [512, 256])
    reward_mode: str = "gain"  # "gain" | "throughput"
    gain_scale: float = 0.25
    throughput_scale: float = 0.01
    gamma: float = 0.97  # discount between cycles ("gain" only)
    gae_lambda: float = 0.9

    @property
    def n_ctx_feats(self):
        return 3 + self.hist_len

    @property
    def depth_feat_dim(self):
        return 4 * self.top_k + self.n_ctx_feats

    @property
    def size_feat_dim(self):
        return self.size_topn + self.n_ctx_feats

    @property
    def cycle_gamma(self):
        return self.gamma if self.reward_mode == "gain" else 0.0


def _mlp(in_dim, hidden, out_dim):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """obs = [hand-made features | normalised g_v]. g_v goes through a learned projection."""

    def __init__(self, feat_dim, vis_in, vis_dim, n_actions, hidden):
        super().__init__()
        self.feat_dim = feat_dim
        self.vis_proj = nn.Sequential(nn.Linear(vis_in, vis_dim), nn.Tanh())
        self.pi = _mlp(feat_dim + vis_dim, hidden, n_actions)
        self.vf = _mlp(feat_dim + vis_dim, hidden, 1)

    def forward(self, obs):
        x = torch.cat(
            (obs[:, : self.feat_dim], self.vis_proj(obs[:, self.feat_dim :])), dim=-1
        )
        return self.pi(x), self.vf(x).squeeze(-1)


class _Draft:
    """Bookkeeping for one draft tree (= one cycle)."""

    __slots__ = (
        "depth_steps", "size_step", "draft_time", "first", "levels", "size",
        "target_depth", "ctx", "prev_tokens",
    )

    def __init__(self, first, ctx, prev_tokens):
        self.depth_steps = []
        self.size_step = None
        self.draft_time = 0.0
        self.first = first
        self.levels = 0
        self.size = 0
        self.target_depth = 0
        self.ctx = ctx  # context length the target sees when verifying this draft
        self.prev_tokens = prev_tokens  # tokens the draft model has to catch up on


class DraftController:
    def __init__(self, cfg: RLConfig, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.depth_net = ActorCritic(
            cfg.depth_feat_dim, cfg.hidden_size, cfg.vis_dim, 2, cfg.pi_hidden
        ).to(self.device)
        self.size_net = ActorCritic(
            cfg.size_feat_dim, cfg.hidden_size, cfg.vis_dim, len(cfg.size_choices), cfg.pi_hidden
        ).to(self.device)
        self.depth_mode = "fixed"
        self.size_mode = "fixed"
        self.train_depth = False
        self.train_size = False
        self.trained = {"depth": False, "size": False}
        self.spec_layer = None

        # timing / reward
        self.measure = False  # True: synchronise CUDA and time every cycle
        self.time_source = "measured"  # "measured" | "model"
        self.cost = CostModel()
        self.rate_ref: Optional[float] = None  # reference throughput (tokens/sec)
        self.log_cycles = False  # keep raw cycle records (calibration)
        self.cycle_log = []
        self.calib_policy_cost = False  # run the policy nets in non-policy modes (calibration)

        # training data
        self.buf = {"depth": [], "size": []}  # (obs, action, logp, advantage, return)
        self._ep = {"depth": [], "size": []}  # trajectory of the current generation
        self._cycle_idx = 0
        self._roll_tok = 0.0
        self._roll_time = 0.0
        self.stats = dict(cycles=0, tokens=0, draft_time=0.0, verify_time=0.0, levels=0, size=0)

        self._drafting: Optional[_Draft] = None
        self._verified: Optional[_Draft] = None
        self.begin_episode(0, 0)

    # ------------------------------------------------------------------ setup
    def attach(self, spec_layer):
        if spec_layer.top_k != self.cfg.top_k:
            raise ValueError(f"policy top_k={self.cfg.top_k} but draft model top_k={spec_layer.top_k}")
        self.spec_layer = spec_layer
        spec_layer.rl_controller = self
        return self

    def set_modes(self, depth_mode=None, size_mode=None, train_depth=None, train_size=None):
        if depth_mode is not None:
            assert depth_mode in MODES
            self.depth_mode = depth_mode
        if size_mode is not None:
            assert size_mode in MODES
            self.size_mode = size_mode
        if train_depth is not None:
            self.train_depth = bool(train_depth) and self.depth_mode == "policy"
        if train_size is not None:
            self.train_size = bool(train_size) and self.size_mode == "policy"
        return self

    @property
    def max_size(self):
        return max(self.cfg.size_choices) - 1

    @property
    def _rewarding(self):
        return self.measure or self.time_source == "model"

    # ------------------------------------------------------- hooks: generation
    def now(self):
        if not self.measure:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def begin_episode(self, ctx_len, n_img):
        """Called once per specgenerate() call, before the first draft."""
        self.finish_episode()
        self.ctx_len = int(ctx_len)
        self.n_img = int(n_img)
        self.hist = deque([-1.0] * self.cfg.hist_len, maxlen=self.cfg.hist_len)
        self._gv = None
        self._drafting = None  # a draft that was never verified is dropped here
        self._verified = None
        self._first = True
        self._last_tokens = 0

    def on_draft_start(self):
        """Start of topK_genrate."""
        self._verified = self._drafting
        self._drafting = _Draft(self._first, self.ctx_len, self._last_tokens)
        self._first = False
        if self.depth_mode == "random":
            self._drafting.target_depth = random.randint(self.cfg.min_depth, self.cfg.max_depth)
        self._t_draft = self.now()

    def loop_depth(self, fixed_depth):
        """Upper bound of the expansion loop for this draft."""
        if self.depth_mode == "fixed":
            return fixed_depth
        if self.depth_mode == "random":
            return self._drafting.target_depth
        return self.cfg.max_depth

    def continue_drafting(self, depth_done, scores, topk_p, last_p):
        """Called after every expansion. Returns False to stop expanding the tree."""
        d = self._drafting
        d.levels = depth_done
        decision_point = self.cfg.min_depth <= depth_done < self.cfg.max_depth
        if self.depth_mode != "policy":
            if self.calib_policy_cost and decision_point:
                # pay the cost of a real decision so that calibration timings include it
                self._act(self.depth_net, self._depth_obs(depth_done, scores, topk_p, last_p), False)
            return True
        if not decision_point:
            return depth_done < self.cfg.max_depth
        obs = self._depth_obs(depth_done, scores, topk_p, last_p)
        a, logp, v = self._act(self.depth_net, obs, self.train_depth)
        if self.train_depth:
            d.depth_steps.append((obs, a, logp, v))
        return a == 1

    def choose_size(self, scores_list, fixed_size):
        """Called once the tree is built. Returns the number of draft tokens to verify."""
        d = self._drafting
        if self.size_mode == "policy":
            obs = self._size_obs(d.levels, scores_list)
            a, logp, v = self._act(self.size_net, obs, self.train_size)
            if self.train_size:
                d.size_step = (obs, a, logp, v)
            size = self.cfg.size_choices[a] - 1
        else:
            if self.calib_policy_cost:
                self._act(self.size_net, self._size_obs(d.levels, scores_list), False)
            size = fixed_size if self.size_mode == "fixed" else random.choice(self.cfg.size_choices) - 1
        d.size = min(int(size), scores_list.shape[0])
        return d.size

    def on_draft_end(self):
        """End of topK_genrate."""
        if self.measure:
            self._drafting.draft_time = self.now() - self._t_draft

    def on_verified(self, n_tokens):
        """After evaluate_posterior: n_tokens = accept_length + 1 tokens were produced."""
        self._last_tokens = int(n_tokens)
        self.hist.append(self._last_tokens / (self.cfg.max_depth + 1))
        self.ctx_len += self._last_tokens

    def end_cycle(self, iter_time):
        """After update_inference_inputs. iter_time = verification + next draft."""
        d = self._verified
        self._verified = None
        if d is None or d.first or not self._rewarding:
            return  # the first draft also encodes the whole prompt: not comparable
        n = self._last_tokens
        if self.measure:
            t_d = d.draft_time
            t_v = max(iter_time - self._drafting.draft_time, 1e-6)
            if self.log_cycles:
                self.cycle_log.append((d.levels, d.size, d.prev_tokens, d.ctx, n, t_d, t_v))
        if self.time_source == "model":
            t_d = self.cost.draft(d.levels, d.size, d.prev_tokens)
            t_v = self.cost.verify(d.size, d.ctx)
        cycle_time = t_d + t_v

        s = self.stats
        s["cycles"] += 1
        s["tokens"] += n
        s["draft_time"] += t_d
        s["verify_time"] += t_v
        s["levels"] += d.levels
        s["size"] += d.size
        if not (self.train_depth or self.train_size):
            return

        self._roll_tok += n
        self._roll_time += cycle_time
        if self.cfg.reward_mode == "gain":
            reward = self.cfg.gain_scale * (n - self.rate_ref * cycle_time)
        else:
            reward = self.cfg.throughput_scale * n / cycle_time

        c = self._cycle_idx
        self._cycle_idx += 1
        if d.depth_steps:
            last = len(d.depth_steps) - 1
            for j, (obs, a, logp, v) in enumerate(d.depth_steps):
                self._ep["depth"].append([obs, a, logp, v, reward if j == last else 0.0, c])
        elif self.train_depth and self._ep["depth"]:
            self._ep["depth"][-1][4] += reward  # cycle without a depth decision
        if d.size_step is not None:
            obs, a, logp, v = d.size_step
            self._ep["size"].append([obs, a, logp, v, reward, c])

    def finish_episode(self):
        """Turn the trajectory of the finished generation into (advantage, return) samples.

        GAE over the decisions of one generation. Inside a cycle the discount is 1,
        between cycles it is cfg.cycle_gamma (0 in "throughput" mode, i.e. every cycle
        is its own episode). The end of a generation is terminal with value 0; with the
        "gain" reward the expected future gain is ~0 by construction of rate_ref.
        """
        g, lam = self.cfg.cycle_gamma, self.cfg.gae_lambda
        for name in ("depth", "size"):
            traj = self._ep[name]
            self._ep[name] = []
            if not traj:
                continue
            vals = torch.stack([s[3] for s in traj]).float().tolist()
            adv, next_v = 0.0, 0.0
            for t in reversed(range(len(traj))):
                obs, a, logp, _, r, c = traj[t]
                if t == len(traj) - 1:
                    disc = 0.0
                else:
                    disc = 1.0 if traj[t + 1][5] == c else g
                delta = r + disc * next_v - vals[t]
                adv = delta + disc * lam * adv
                self.buf[name].append((obs, a, logp, adv, adv + vals[t]))
                next_v = vals[t]

    def update_rate(self, ema=0.5):
        """Refresh the reference throughput from the cycles of the last rollout."""
        if self._roll_time > 0:
            new = self._roll_tok / self._roll_time
            self.rate_ref = new if self.rate_ref is None else (1 - ema) * self.rate_ref + ema * new
        self._roll_tok = 0.0
        self._roll_time = 0.0
        return self.rate_ref

    # ------------------------------------------------------------- features
    def _act(self, net, obs, explore):
        logits, value = net(obs.unsqueeze(0))
        if explore:
            dist = Categorical(logits=logits)
            a = dist.sample()
            return int(a), dist.log_prob(a).squeeze(0), value.squeeze(0)
        return int(logits.argmax(-1)), None, None

    def _visual(self):
        if self._gv is None:
            h = self.cfg.hidden_size
            g = getattr(self.spec_layer, "last_img_hidden", None) if self.cfg.use_visual else None
            if g is None or g.numel() != h:
                self._gv = torch.zeros(h, device=self.device)
            else:
                g = g.detach().reshape(-1).to(self.device, torch.float32)
                self._gv = g / (g.norm() + 1e-6) * math.sqrt(h)
        return self._gv

    def _ctx_feats(self, depth_frac):
        ctx = max(self.ctx_len, 1)
        img_frac = self.n_img / ctx if self.cfg.use_visual else 0.0
        vals = [depth_frac, min(ctx / 4096.0, 2.0), img_frac] + list(self.hist)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def _depth_obs(self, depth_done, scores, topk_p, last_p):
        # scores : [top_k]         cumulative log-prob of the new frontier
        # topk_p : [top_k, top_k]  local top-k log-probs of the nodes just expanded
        # last_p : [top_k, vocab]  full log-softmax of the nodes just expanded
        f_scores = scores.float().clamp(-20, 0) / 10 + 1
        p = topk_p.float().exp()
        lp = last_p.float().clamp_min(-50)
        f_ent = -(lp.exp() * lp).sum(-1) / 10
        return torch.cat(
            (
                f_scores.to(self.device),
                p[:, 0].to(self.device),  # top-1 probability
                (p[:, 0] - p[:, 1]).to(self.device),  # margin
                f_ent.to(self.device),  # entropy
                self._ctx_feats(depth_done / self.cfg.max_depth),
                self._visual(),
            )
        )

    def _size_obs(self, levels, scores_list):
        n = min(self.cfg.size_topn, scores_list.shape[0])
        top = torch.topk(scores_list.float(), n).values.clamp(-20, 0) / 10 + 1
        if n < self.cfg.size_topn:
            top = torch.cat((top, top.new_full((self.cfg.size_topn - n,), -1.0)))
        return torch.cat(
            (top.to(self.device), self._ctx_feats(levels / self.cfg.max_depth), self._visual())
        )

    # ------------------------------------------------------------------- io
    def summary(self):
        s = self.stats
        c = max(s["cycles"], 1)
        t = s["draft_time"] + s["verify_time"]
        return {
            "cycles": s["cycles"],
            "tokens_per_sec": s["tokens"] / t if t > 0 else 0.0,
            "tokens_per_cycle": s["tokens"] / c,
            "draft_ms_per_cycle": 1000 * s["draft_time"] / c,
            "verify_ms_per_cycle": 1000 * s["verify_time"] / c,
            "mean_depth": s["levels"] / c,
            "mean_size": s["size"] / c,
            "depth_mode": self.depth_mode,
            "size_mode": self.size_mode,
            "time_source": self.time_source,
        }

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0 if isinstance(self.stats[k], int) else 0.0

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "depth_net": self.depth_net.state_dict(),
                "size_net": self.size_net.state_dict(),
                "trained": dict(self.trained),
                "cost": self.cost.state(),
                "rate_ref": self.rate_ref,
            },
            path,
        )

    @classmethod
    def load(cls, path, device):
        ck = torch.load(path, map_location="cpu")
        ctrl = cls(RLConfig(**ck["cfg"]), device)
        ctrl.depth_net.load_state_dict(ck["depth_net"])
        ctrl.size_net.load_state_dict(ck["size_net"])
        ctrl.trained = dict(ck.get("trained", {"depth": True, "size": True}))
        ctrl.cost = CostModel.from_state(ck.get("cost"))
        ctrl.rate_ref = ck.get("rate_ref")
        return ctrl


def attach_from_env(model, spec):
    """Used by SpecModel.from_pretrained when VISPEC_RL_POLICY is set.

    VISPEC_RL_POLICY=<ckpt.pt>  use the trained policies (greedy actions)
    VISPEC_RL_POLICY=fixed      keep ViSpec's fixed depth/size (for profiling only)
    VISPEC_RL_DEPTH_MODE / VISPEC_RL_SIZE_MODE = fixed|policy   override (ablations)
    VISPEC_RL_PROFILE=<out.json>  time every cycle and dump a summary at exit
    """
    device = model.spec_layer.embed_tokens.weight.device
    if spec == "fixed":
        ctrl = DraftController(
            RLConfig(top_k=model.spec_layer.top_k, hidden_size=model.hidden_size), device
        )
    else:
        ctrl = DraftController.load(spec, device)
        if ctrl.cfg.hidden_size != model.hidden_size:
            raise ValueError("RL policy was trained for a different target model (hidden size differs)")
        ctrl.set_modes(
            depth_mode=os.environ.get(
                "VISPEC_RL_DEPTH_MODE", "policy" if ctrl.trained.get("depth") else "fixed"
            ),
            size_mode=os.environ.get(
                "VISPEC_RL_SIZE_MODE", "policy" if ctrl.trained.get("size") else "fixed"
            ),
        )
    ctrl.depth_net.eval()
    ctrl.size_net.eval()
    ctrl.attach(model.spec_layer)
    prof = os.environ.get("VISPEC_RL_PROFILE")
    if prof:
        ctrl.measure = True

        def _dump():
            with open(prof, "w") as f:
                json.dump(ctrl.summary(), f, indent=2)

        atexit.register(_dump)
    print(f"[vispec.rl] controller attached: depth={ctrl.depth_mode} size={ctrl.size_mode}")
    return ctrl
