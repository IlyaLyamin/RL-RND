"""Обучение PPO на одной карте MiniGrid.

Внешняя программа вызывает :func:`train` отдельно для каждой карты
и сама выбирает папку запуска. Функция возвращает время до остановки
и успех: не меньше ``N_SUCCESS`` попаданий в цель среди последних
``SUCCESS_WINDOW`` эпизодов.
"""

# Каждый N-й завершённый эпизод пишется в консоль.
LOG_EVERY_EPISODES = 100
# Длина вектора признаков свёрточной сети.
FEATURES_DIM = 128
# В каналах лежат индексы объектов, а не яркость пикселей.
NORMALIZE_IMAGES = False
# Параметры PPO.
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
# Свёртки для картинки MiniGrid.
CNN_CHANNELS = (16, 32, 64)
CNN_KERNEL = 2
# Успех запуска.
N_SUCCESS = 8
SUCCESS_WINDOW = 10
# Имя папки запуска: desktop|laptop|server, затем cpu|cuda, затем ver и номер.
RUN_DIR_PATTERN = r"^(desktop|laptop|server)_(cpu|cuda)_ver(\d+)$"
# Обязательные ключи. Лишний или пропущенный ключ - ошибка.
FIXED_KEYS = frozenset({"max_episode_steps", "total_timesteps", "no_improve_episodes"})
CHANGABLE_KEYS = frozenset({"device"})

import logging
import re
import time
from pathlib import Path

import gymnasium as gym
import minigrid
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

RUN_DIR_NAME = re.compile(RUN_DIR_PATTERN)

logger = logging.getLogger("train")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


def episode_success(goal_hits: list[bool]) -> bool:
    """Успех, если в последних SUCCESS_WINDOW эпизодах не меньше N_SUCCESS целей."""
    if len(goal_hits) < SUCCESS_WINDOW:
        return False
    return sum(goal_hits[-SUCCESS_WINDOW:]) >= N_SUCCESS


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Свёртки для символической картинки MiniGrid 7×7."""

    def __init__(self, observation_space: gym.Space, features_dim: int = FEATURES_DIM) -> None:
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]
        layers: list[nn.Module] = []
        for out_channels in CNN_CHANNELS:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=CNN_KERNEL))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class FpsWriter(KVWriter):
    """Пишет в fps.txt пару «шаг fps» на каждом отчёте PPO."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", encoding="utf-8")

    def write(
        self,
        key_values: dict[str, object],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        fps = key_values.get("time/fps")
        if fps is None:
            return
        self._file.write(f"{step} {fps}\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class EpisodeStats(BaseCallback):
    """Логи эпизодов, лучшая модель и остановка без роста награды."""

    def __init__(self, map_dir: Path, no_improve_episodes: int) -> None:
        super().__init__()
        self._map_dir = map_dir
        self._no_improve_episodes = no_improve_episodes
        self._reward_file = (map_dir / "reward.txt").open("w", encoding="utf-8")
        self._steps_file = (map_dir / "steps_per_ep.txt").open("w", encoding="utf-8")
        self._best_reward = -float("inf")
        self._episodes_without_improvement = 0
        self._episode = 0
        self._goal_hits: list[bool] = []

    def _on_step(self) -> bool:
        if not bool(self.locals["dones"][0]):
            return True
        episode_info = self.locals["infos"][0].get("episode")
        if episode_info is None:
            return True

        reward = float(episode_info["r"])
        steps = int(episode_info["l"])
        self._episode += 1
        self._reward_file.write(f"{self._episode} {reward:.6f}\n")
        self._steps_file.write(f"{self._episode} {steps}\n")
        self._reward_file.flush()
        self._steps_file.flush()

        self._goal_hits.append(reward > 0)
        if self._episode % LOG_EVERY_EPISODES == 0:
            logger.info("Эпизод %s завершён, reward %.6f", self._episode, reward)

        if reward > self._best_reward:
            self._best_reward = reward
            self.model.save(str(self._map_dir / "best_model"))
            self._episodes_without_improvement = 0
            return True

        self._episodes_without_improvement += 1
        if self._episodes_without_improvement < self._no_improve_episodes:
            return True
        logger.info(
            "Остановка: награда не росла %s эпизодов. Лучшая награда %.6f",
            self._no_improve_episodes,
            self._best_reward,
        )
        return False

    def close(self) -> None:
        if not self._reward_file.closed:
            self._reward_file.close()
        if not self._steps_file.closed:
            self._steps_file.close()

    @property
    def success(self) -> bool:
        return episode_success(self._goal_hits)


def _require_map_name(map_name: object) -> str:
    """Проверяет, что передано полное имя зарегистрированной карты."""
    if type(map_name) is not str or map_name == "":
        raise TypeError("map_name должен быть непустой строкой с полным id карты")
    _ = minigrid.__version__
    try:
        gym.spec(map_name)
    except gym.error.Error as exc:
        raise ValueError(f"неизвестная карта {map_name!r}") from exc
    return map_name


def _require_run_dir(run_dir: object) -> str:
    """Проверяет имя папки запуска."""
    if type(run_dir) is not str or run_dir == "":
        raise TypeError("run_dir должен быть непустой строкой")
    if RUN_DIR_NAME.fullmatch(Path(run_dir).name) is None:
        raise ValueError(
            "имя папки должно иметь вид "
            "<desktop/laptop/server>_<cpu/cuda>_ver<номер>"
        )
    return run_dir


def _key_error(name: str, keys: set[str], expected: frozenset[str]) -> None:
    """Сообщает о лишних и пропущенных ключах."""
    extra = sorted(keys - expected)
    missing = sorted(expected - keys)
    parts: list[str] = []
    if extra:
        parts.append(f"лишние {extra}")
    if missing:
        parts.append(f"отсутствуют {missing}")
    raise ValueError(f"{name}: {', '.join(parts)}")


def _require_positive_int(source: str, key: str, value: object) -> int:
    """Проверяет, что значение — целое число больше нуля."""
    if type(value) is not int or value < 1:
        raise TypeError(f"{source}[{key!r}] должен быть целым числом >= 1")
    return value


def _require_fixed(fixed_params: object) -> dict[str, int]:
    """Проверяет набор параметров, общий для всех карт одного запуска."""
    if type(fixed_params) is not dict:
        raise TypeError("fixed_params должен быть dict")
    keys = set(fixed_params)
    if keys != FIXED_KEYS:
        _key_error("fixed_params", keys, FIXED_KEYS)
    return {
        key: _require_positive_int("fixed_params", key, fixed_params[key])
        for key in FIXED_KEYS
    }


def _require_device(changable_params: object, run_dir: str) -> str:
    """Проверяет устройство и совпадение с cpu/cuda в имени папки."""
    if type(changable_params) is not dict:
        raise TypeError("changable_params должен быть dict")
    keys = set(changable_params)
    if keys != CHANGABLE_KEYS:
        _key_error("changable_params", keys, CHANGABLE_KEYS)
    device = changable_params["device"]
    if type(device) is not str or device not in {"cpu", "cuda"}:
        raise TypeError("changable_params['device'] должен быть 'cpu' или 'cuda'")
    folder_device = RUN_DIR_NAME.fullmatch(Path(run_dir).name).group(2)
    if device != folder_device:
        raise ValueError(
            f"device {device!r} не совпадает с папкой {Path(run_dir).name!r}"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda', но PyTorch не видит видеокарту")
    return device


def _require_seed(seed: object) -> int:
    """Проверяет зерно генератора. Ноль допустим."""
    if type(seed) is not int or seed < 0:
        raise TypeError("seed должен быть целым числом >= 0")
    return seed


def _build_model(env: gym.Env, device: str, seed: int) -> PPO:
    """Собирает PPO с свёрточной сетью MiniGrid."""
    return PPO(
        "CnnPolicy",
        env,
        seed=seed,
        device=device,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        policy_kwargs={
            "features_extractor_class": MinigridFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
            "normalize_images": NORMALIZE_IMAGES,
        },
        verbose=0,
    )


def train(
    map_name: str,
    run_dir: str,
    fixed_params: dict[str, int],
    changable_params: dict[str, str],
    seed: int,
) -> tuple[float, bool]:
    """Обучает PPO на одной карте и пишет логи в ``run_dir/map_name``.

    :param map_name: полный id карты, например ``MiniGrid-Empty-8x8-v0``
    :param run_dir: папка запуска, имя вида ``laptop_cuda_ver0``
    :param fixed_params: ``max_episode_steps``, ``total_timesteps``, ``no_improve_episodes``
    :param changable_params: ``device`` — ``cpu`` или ``cuda``
    :param seed: общее зерно для всех карт и машин
    :return: время до остановки в секундах и успех по последним эпизодам
    """
    map_name = _require_map_name(map_name)
    run_dir = _require_run_dir(run_dir)
    fixed = _require_fixed(fixed_params)
    device = _require_device(changable_params, run_dir)
    seed = _require_seed(seed)

    map_dir = Path(run_dir) / map_name
    map_dir.mkdir(parents=True, exist_ok=True)
    stats = EpisodeStats(map_dir, fixed["no_improve_episodes"])
    fps_writer = FpsWriter(map_dir / "fps.txt")
    env: gym.Env | None = None
    model: PPO | None = None
    try:
        env = ImgObsWrapper(gym.make(map_name, max_steps=fixed["max_episode_steps"]))
        model = _build_model(env, device, seed)
        model.set_logger(Logger(folder=str(map_dir), output_formats=[fps_writer]))
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.learn(total_timesteps=fixed["total_timesteps"], callback=stats)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return elapsed, stats.success
    finally:
        stats.close()
        if model is not None:
            model.logger.close()
        else:
            fps_writer.close()
        if env is not None:
            env.close()
