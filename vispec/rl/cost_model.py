"""Linear cost model for one draft-and-verify cycle.

Measured wall-clock time of a single cycle (tens of ms) is noisy, which makes the
reward noisy. The cost model is fitted once on calibration cycles and can replace the
measured time during training (--time-source model). Final speed numbers must always
be measured, never taken from this model.

    draft time  ~ w0 + w1 * levels + w2 * size + w3 * prev_tokens
    verify time ~ v0 + v1 * size   + v2 * ctx_len / 1000
"""

import torch


class CostModel:
    def __init__(self, draft_w=None, verify_w=None):
        self.draft_w = draft_w
        self.verify_w = verify_w

    @property
    def fitted(self):
        return self.draft_w is not None and self.verify_w is not None

    @staticmethod
    def _xd(levels, size, prev_tokens):
        return [1.0, float(levels), float(size), float(prev_tokens)]

    @staticmethod
    def _xv(size, ctx_len):
        return [1.0, float(size), ctx_len / 1000.0]

    def draft(self, levels, size, prev_tokens):
        x = self._xd(levels, size, prev_tokens)
        return max(sum(w * v for w, v in zip(self.draft_w, x)), 1e-5)

    def verify(self, size, ctx_len):
        x = self._xv(size, ctx_len)
        return max(sum(w * v for w, v in zip(self.verify_w, x)), 1e-5)

    def cycle(self, levels, size, prev_tokens, ctx_len):
        return self.draft(levels, size, prev_tokens) + self.verify(size, ctx_len)

    def fit(self, records):
        """records: (levels, size, prev_tokens, ctx_len, n_tokens, t_draft, t_verify)."""
        if len(records) < 16:
            raise ValueError("not enough calibration cycles to fit the cost model")

        def _fit(X, y):
            X = torch.tensor(X, dtype=torch.float64)
            y = torch.tensor(y, dtype=torch.float64)
            w = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
            res = ((X @ w - y) ** 2).sum()
            tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)
            return w.tolist(), float(1 - res / tot)

        self.draft_w, r2_d = _fit(
            [self._xd(r[0], r[1], r[2]) for r in records], [r[5] for r in records]
        )
        self.verify_w, r2_v = _fit(
            [self._xv(r[1], r[3]) for r in records], [r[6] for r in records]
        )
        return {"n": len(records), "r2_draft": r2_d, "r2_verify": r2_v}

    def state(self):
        return {"draft_w": self.draft_w, "verify_w": self.verify_w}

    @classmethod
    def from_state(cls, s):
        return cls(**s) if s else cls()
