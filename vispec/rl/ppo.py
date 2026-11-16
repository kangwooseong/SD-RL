"""Minimal PPO for the two draft policies (no extra dependencies).

Advantages and returns are computed by DraftController.finish_episode (GAE over the
cycles of one generation); this file only does the clipped policy / value update.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOTrainer:
    def __init__(
        self,
        ctrl,
        lr=3e-4,
        clip=0.2,
        epochs=4,
        minibatch=256,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.ctrl = ctrl
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.vf_coef, self.ent_coef, self.max_grad_norm = vf_coef, ent_coef, max_grad_norm
        self.nets = {"depth": ctrl.depth_net, "size": ctrl.size_net}
        self.opts = {k: torch.optim.Adam(n.parameters(), lr=lr) for k, n in self.nets.items()}

    def num_ready(self):
        """Number of collected cycles (size decisions) or depth decisions."""
        b = self.ctrl.buf
        return len(b["size"]) if self.ctrl.train_size else len(b["depth"])

    def update(self):
        self.ctrl.finish_episode()
        out = {}
        for name in ("depth", "size"):
            train = self.ctrl.train_depth if name == "depth" else self.ctrl.train_size
            data = self.ctrl.buf[name]
            self.ctrl.buf[name] = []
            if not train or len(data) < 8:
                continue
            out.update(self._update_one(name, data))
            self.ctrl.trained[name] = True
        return out

    def _update_one(self, name, data):
        net, opt = self.nets[name], self.opts[name]
        dev = self.ctrl.device
        obs = torch.stack([d[0] for d in data]).to(dev).float()
        act = torch.tensor([d[1] for d in data], device=dev, dtype=torch.long)
        logp_old = torch.stack([d[2] for d in data]).to(dev).float().detach()
        adv_raw = torch.tensor([d[3] for d in data], device=dev, dtype=torch.float32)
        ret = torch.tensor([d[4] for d in data], device=dev, dtype=torch.float32)
        adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

        n = obs.shape[0]
        net.train()
        with torch.enable_grad():
            for _ in range(self.epochs):
                perm = torch.randperm(n, device=dev)
                for s in range(0, n, self.minibatch):
                    idx = perm[s : s + self.minibatch]
                    logits, value = net(obs[idx])
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(act[idx])
                    ratio = (logp - logp_old[idx]).exp()
                    surr = torch.min(
                        ratio * adv[idx],
                        ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx],
                    )
                    pi_loss = -surr.mean()
                    v_loss = F.mse_loss(value, ret[idx])
                    ent = dist.entropy().mean()
                    loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                    opt.step()
        net.eval()
        counts = torch.bincount(act, minlength=logits.shape[-1]).float()
        return {
            f"{name}/n": n,
            f"{name}/return": ret.mean().item(),
            f"{name}/pi_loss": pi_loss.item(),
            f"{name}/v_loss": v_loss.item(),
            f"{name}/entropy": ent.item(),
            f"{name}/action_dist": [round(x, 3) for x in (counts / n).tolist()],
        }
