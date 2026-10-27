"""PPO over the controller's parameters: a drop-in replacement for pycma's ask / tell (decision 99).

The policy is a diagonal Gaussian over the same unit-cube parameters CMA-ES searches. Each
generation ``ask`` samples ``popsize`` parameter sets (clipped to [0, 1] for evaluation); ``tell``
takes their losses (minus the episode objective, as for CMA-ES), standardises them into
advantages and takes ``epochs`` Adam steps on PPO's clipped surrogate for the mean and log std.
There is one action per episode, so this is PPO as a policy-gradient search over a fixed
parameter set: the same search space, objective, scenarios and budget as CMA-ES.
"""
from __future__ import annotations

import numpy as np


class PPOSearch:
    def __init__(self, x0, sigma0: float, *, popsize: int, seed: int, lr: float = 0.02, epochs: int = 10,
                 clip: float = 0.2, min_sigma: float = 0.005) -> None:
        self.mu = np.asarray(x0, dtype=float)
        self.log_sigma = np.full_like(self.mu, np.log(sigma0))
        self.popsize, self.lr, self.epochs, self.clip, self.min_log_sigma = popsize, lr, epochs, clip, np.log(min_sigma)
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros(2 * self.mu.size)
        self.v = np.zeros(2 * self.mu.size)
        self.t = 0
        self._samples = None

    @property
    def mean(self) -> np.ndarray:
        return np.clip(self.mu, 0.0, 1.0)

    @property
    def sigma(self) -> float:
        return float(np.exp(self.log_sigma).mean())

    def stop(self) -> dict:
        return {}

    def _logp(self, x):
        z = (x - self.mu) / np.exp(self.log_sigma)
        return (-0.5 * z ** 2 - self.log_sigma).sum(-1)

    def ask(self) -> list[np.ndarray]:
        self._samples = self.mu + np.exp(self.log_sigma) * self.rng.standard_normal((self.popsize, self.mu.size))
        self._old_logp = self._logp(self._samples)
        return [np.clip(x, 0.0, 1.0) for x in self._samples]

    def tell(self, solutions, losses) -> None:
        x, rewards = self._samples, -np.asarray(losses, dtype=float)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        for _ in range(self.epochs):
            sigma = np.exp(self.log_sigma)
            ratio = np.exp(self._logp(x) - self._old_logp)
            # Gradient of the clipped surrogate: only samples whose unclipped term is the minimum.
            active = ~(((adv > 0) & (ratio > 1 + self.clip)) | ((adv < 0) & (ratio < 1 - self.clip)))
            weight = (ratio * adv * active)[:, None] / len(adv)
            z = (x - self.mu) / sigma
            grad = np.concatenate([(weight * z / sigma).sum(0), (weight * (z ** 2 - 1)).sum(0)])  # ascent direction
            self.t += 1
            self.m = 0.9 * self.m + 0.1 * grad
            self.v = 0.999 * self.v + 0.001 * grad ** 2
            step = self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
            self.mu = self.mu + step[: self.mu.size]
            self.log_sigma = np.maximum(self.log_sigma + step[self.mu.size:], self.min_log_sigma)
