"""gym.Wrapper, добавляющий β · r_int к внешней награде."""

import gymnasium as gym
import numpy as np

from .rnd_module import RNDModule


class IntrinsicRewardWrapper(gym.Wrapper):
    """Считает r_int на каждом шаге и отдаёт SB3 сумму r_ext + β · r_int.

    Должен стоять **после** ``ImgObsWrapper`` (obs = (C,H,W)) и **до** SB3 ``VecEnv``.
    Дополнительные поля кладутся в ``info``:
    ``r_ext``, ``r_int``, ``r_int_scaled``, ``rnd_loss``, ``beta``.
    """

    def __init__(
        self,
        env: gym.Env,
        rnd_module: RNDModule,
        beta: float,
        beta_decay: float = 1.0,
        beta_min: float = 0.0,
        update_every: int = 1,
    ) -> None:
        super().__init__(env)
        if beta < 0:
            raise ValueError("beta должен быть >= 0")
        if not (0.0 < beta_decay <= 1.0):
            raise ValueError("beta_decay должен быть в (0, 1]")
        if beta_min < 0:
            raise ValueError("beta_min должен быть >= 0")
        if update_every < 1:
            raise ValueError("update_every должен быть >= 1")
        self.rnd_module = rnd_module
        self.beta = float(beta)
        self.beta_decay = float(beta_decay)
        self.beta_min = float(beta_min)
        self.update_every = int(update_every)
        self._step_counter = 0

    def step(self, action):  # type: ignore[override]
        obs, r_ext, terminated, truncated, info = self.env.step(action)
        r_int = self.rnd_module.intrinsic_reward(obs)

        self._step_counter += 1
        if self._step_counter % self.update_every == 0:
            rnd_loss = self.rnd_module.update(obs)
        else:
            rnd_loss = float("nan")

        beta_used = self.beta
        r = float(r_ext) + beta_used * float(r_int)

        # Затухание применяем после использования beta на этом шаге.
        self.beta = max(self.beta_min, self.beta * self.beta_decay)

        info = dict(info)
        info["r_ext"] = float(r_ext)
        info["r_int"] = float(r_int)
        info["r_int_scaled"] = beta_used * float(r_int)
        info["rnd_loss"] = rnd_loss
        info["beta"] = beta_used
        return obs, r, terminated, truncated, info

    def reset(self, **kwargs):  # type: ignore[override]
        return self.env.reset(**kwargs)
