"""Target/predictor сети RND и нормализация входа/награды."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .running_mean_std import RunningMeanStd

CNN_CHANNELS = (16, 32, 64)
CNN_KERNEL = 2
_EPS = 1e-8


class _Encoder(nn.Module):
    """CNN-энкодер, устойчивый к любому H, W >= 1.

    Одна свёртка с padding=1 (размер H, W не уменьшается) →
    AdaptiveAvgPool2d(1) → Flatten → Linear. Работает на obs
    любого размера: 5×5, 6×2, 7×7, 11×11, 16×16, ...
    """

    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, CNN_CHANNELS[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(CNN_CHANNELS[0], CNN_CHANNELS[-1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.linear = nn.Linear(CNN_CHANNELS[-1], feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class RNDModule:
    """RND: target (frozen, random) и predictor (trainable).

    :param obs_shape: форма obs после ``ImgObsWrapper``, например ``(3, 7, 7)``
    :param feature_dim: размер выходного признака обеих сетей
    :param lr: learning rate Adam для predictor
    :param device: ``cpu`` или ``cuda``; для MiniGrid хватает ``cpu``
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        feature_dim: int = 128,
        lr: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        if len(obs_shape) != 3:
            raise ValueError(f"obs_shape должен быть (C,H,W) или (H,W,C), получено {obs_shape!r}")
        # MiniGrid после ImgObsWrapper отдаёт HWC (7,7,3); SB3 CnnPolicy
        # транспонирует в CHW через VecTransposeImage. RNDModule делает то же сам.
        self.channels_last = obs_shape[-1] <= 16 and obs_shape[0] > 16
        if self.channels_last:
            h, w, c = obs_shape
            self.obs_shape = (c, h, w)
        else:
            self.obs_shape = tuple(int(d) for d in obs_shape)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)

        self.target = _Encoder(self.obs_shape[0], self.feature_dim).to(self.device)
        self.predictor = _Encoder(self.obs_shape[0], self.feature_dim).to(self.device)

        for param in self.target.parameters():
            param.requires_grad_(False)
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd(self.obs_shape)   # статистики в CHW
        self.reward_rms = RunningMeanStd(())

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Приводит numpy-obs (HWC или CHW) к нормализованному тензору (1, C, H, W)."""
        arr = np.asarray(obs, dtype=np.float32)
        if self.channels_last:
            arr = np.transpose(arr, (2, 0, 1))   # HWC -> CHW
        norm = self.obs_rms.normalize(arr).astype(np.float32)
        return torch.as_tensor(norm, device=self.device).unsqueeze(0)

    def _to_chw_np(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float64)
        if self.channels_last:
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    @torch.no_grad()
    def intrinsic_reward(self, obs: np.ndarray) -> float:
        """r_int = err / sqrt(var_err + eps), err — MSE признаков target/predictor."""
        chw = self._to_chw_np(obs)
        self.obs_rms.update(chw)
        norm = self.obs_rms.normalize(chw).astype(np.float32)
        tensor = torch.as_tensor(norm, device=self.device).unsqueeze(0)
        err = ((self.target(tensor) - self.predictor(tensor)) ** 2).mean().item()
        self.reward_rms.update(np.asarray(err, dtype=np.float64))
        # Защита от выброса на первых шагах, когда var_err ещё ~0.
        normalized = err / float(np.sqrt(self.reward_rms.var + _EPS))
        return float(np.clip(normalized, -5.0, 5.0))

    def update(self, obs: np.ndarray) -> float:
        chw = self._to_chw_np(obs)
        norm = self.obs_rms.normalize(chw).astype(np.float32)
        tensor = torch.as_tensor(norm, device=self.device).unsqueeze(0)
        target_feat = self.target(tensor)
        pred_feat = self.predictor(tensor)
        loss = ((target_feat - pred_feat) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        """Сохраняет веса и статистики (для чекпоинтов и отладки)."""
        payload = {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "obs_shape": self.obs_shape,
            "feature_dim": self.feature_dim,
        }
        torch.save(payload, str(path))

    def load(self, path: str | Path) -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.target.load_state_dict(payload["target"])
        self.predictor.load_state_dict(payload["predictor"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.obs_rms.load_state_dict(payload["obs_rms"])
        self.reward_rms.load_state_dict(payload["reward_rms"])
