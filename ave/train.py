"""Обучение PPO на одной карте MiniGrid.

Внешняя программа вызывает :func:`train` отдельно для каждой карты
и сама выбирает папку запуска. Функция возвращает время до остановки
и успех: не меньше ``n_success`` попаданий в цель среди последних
``success_window`` эпизодов.

Путь B (RND): если в ``fixed_params`` есть ``use_rnd=True``, среда
оборачивается в :class:`IntrinsicRewardWrapper`, который добавляет
``β · r_int`` к внешней награде. PPO учится как обычно, одной
value-головой. Baseline — при ``use_rnd=False`` (по умолчанию).
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
# Имя папки запуска: desktop|laptop|server, затем cpu|cuda, затем ver и номер.
RUN_DIR_PATTERN = r"^(desktop|laptop|server)_(cpu|cuda)_ver(\d+)$"
# Обязательные ключи. Лишний или пропущенный ключ - ошибка.
FIXED_KEYS = frozenset({
    "max_episode_steps",
    "total_timesteps",
    "no_improve_episodes",
    "n_success",
    "success_window",
    "reward_delta",
})
CHANGABLE_KEYS = frozenset({"device"})
# Необязательные ключи RND. Если в fixed_params нет ни одного — RND выключен.
RND_KEYS = frozenset({
    "use_rnd",
    "rnd_feature_dim",
    "rnd_lr",
    "rnd_beta",
    "rnd_beta_decay",
    "rnd_beta_min",
    "rnd_update_every",
})
RND_DEFAULTS: dict[str, object] = {
    "use_rnd": False,
    "rnd_feature_dim": 128,
    "rnd_lr": 1e-4,
    "rnd_beta": 0.1,
    "rnd_beta_decay": 1.0,
    "rnd_beta_min": 0.0,
    "rnd_update_every": 1,
}

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

from src.rnd import IntrinsicRewardWrapper, RNDModule

RUN_DIR_NAME = re.compile(RUN_DIR_PATTERN)

logger = logging.getLogger("train")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


def recent_goals(goal_hits: list[bool], success_window: int) -> tuple[int, int]:
    """Сколько эпизодов из последних success_window закончились целью, и сколько их было."""
    tail = goal_hits[-success_window:]
    return sum(tail), len(tail)


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
    """Логи эпизодов, лучшая модель и остановка без роста награды.

    При ``use_rnd=True``:
    - дополнительно пишет ``intrinsic_reward.txt``:
      ``эпизод Σ r_int_scaled средний rnd_loss beta``;
    - ``best_model`` и остановка считаются по **внешней** награде
      ``r_ext`` (сумма за эпизод), а не по ``r_ext + β·r_int``.
      Иначе novelty-выброс в первом эпизоде «залипает» как лучшая
      модель и остановка срабатывает формально.

    При ``use_rnd=False`` поведение идентично исходному: ``best_model``
    и остановка по ``reward`` эпизода.
    """

    def __init__(self, map_dir: Path, fixed: dict[str, int | float | bool]) -> None:
        super().__init__()
        self._map_dir = map_dir
        self._no_improve_episodes = int(fixed["no_improve_episodes"])
        self._reward_delta = float(fixed["reward_delta"])
        self._success_window = int(fixed["success_window"])
        self._use_rnd = bool(fixed.get("use_rnd", False))
        self._reward_file = (map_dir / "reward.txt").open("w", encoding="utf-8")
        self._steps_file = (map_dir / "steps_per_ep.txt").open("w", encoding="utf-8")
        self._intrinsic_file = (
            (map_dir / "intrinsic_reward.txt").open("w", encoding="utf-8")
            if self._use_rnd
            else None
        )
        self._best_reward = -float("inf")
        self._episodes_without_improvement = 0
        self._episode = 0
        self._goal_hits: list[bool] = []
        # Аккумуляторы за текущий эпизод.
        self._ep_int_sum = 0.0
        self._ep_loss_sum = 0.0
        self._ep_loss_count = 0
        self._ep_beta_last = 0.0
        self._ep_r_ext_sum = 0.0

    def _on_step(self) -> bool:
        info0 = self.locals["infos"][0]

        # Накапливаем r_ext (для best_model/остановки в RND-режиме)
        # и поля RND (для intrinsic_reward.txt).
        r_ext = info0.get("r_ext")
        if r_ext is not None:
            self._ep_r_ext_sum += float(r_ext)
        if self._intrinsic_file is not None:
            r_int_scaled = info0.get("r_int_scaled")
            if r_int_scaled is not None:
                self._ep_int_sum += float(r_int_scaled)
            loss = info0.get("rnd_loss")
            if loss is not None and not (isinstance(loss, float) and loss != loss):
                self._ep_loss_sum += float(loss)
                self._ep_loss_count += 1
            beta = info0.get("beta")
            if beta is not None:
                self._ep_beta_last = float(beta)

        if not bool(self.locals["dones"][0]):
            return True
        episode_info = info0.get("episode")
        if episode_info is None:
            return True

        reward = float(episode_info["r"])
        steps = int(episode_info["l"])
        self._episode += 1
        self._reward_file.write(f"{self._episode} {reward:.6f}\n")
        self._steps_file.write(f"{self._episode} {steps}\n")
        self._reward_file.flush()
        self._steps_file.flush()

        if self._intrinsic_file is not None:
            mean_loss = (
                self._ep_loss_sum / self._ep_loss_count if self._ep_loss_count else 0.0
            )
            self._intrinsic_file.write(
                f"{self._episode} {self._ep_int_sum:.6f} "
                f"{mean_loss:.6f} {self._ep_beta_last:.6f}\n"
            )
            self._intrinsic_file.flush()
            self._ep_int_sum = 0.0
            self._ep_loss_sum = 0.0
            self._ep_loss_count = 0

        # Для best_model/остановки: в RND — внешняя награда, иначе — reward.
        metric = self._ep_r_ext_sum if self._use_rnd else reward
        self._ep_r_ext_sum = 0.0

        self._goal_hits.append(reward > 0)
        if self._episode % LOG_EVERY_EPISODES == 0:
            logger.info("Эпизод %s завершён, reward %.6f", self._episode, reward)

        if metric > self._best_reward + self._reward_delta:
            self._best_reward = metric
            self.model.save(str(self._map_dir / "best_model"))
            self._episodes_without_improvement = 0
            return True

        self._episodes_without_improvement += 1
        if self._episodes_without_improvement < self._no_improve_episodes:
            return True
        logger.info(
            "Остановка: награда не выросла больше чем на %s за %s эпизодов. Лучшая награда %.6f",
            self._reward_delta,
            self._no_improve_episodes,
            self._best_reward,
        )
        return False

    def close(self) -> None:
        if not self._reward_file.closed:
            self._reward_file.close()
        if not self._steps_file.closed:
            self._steps_file.close()
        if self._intrinsic_file is not None and not self._intrinsic_file.closed:
            self._intrinsic_file.close()

    def goal_count(self) -> tuple[int, int]:
        return recent_goals(self._goal_hits, self._success_window)


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


def _require_delta(value: object) -> float:
    """Проверяет порог роста награды. Ноль допустим."""
    if type(value) not in (int, float) or isinstance(value, bool) or value < 0:
        raise TypeError("fixed_params['reward_delta'] должен быть числом >= 0")
    return float(value)


def _require_fixed(fixed_params: object) -> dict[str, int | float | bool]:
    """Проверяет параметры запуска. Ключи RND опциональны.

    Обязательны ``FIXED_KEYS``; допустимы также ``RND_KEYS`` (с дефолтами
    из ``RND_DEFAULTS``). Любой другой ключ — ошибка.
    """
    if type(fixed_params) is not dict:
        raise TypeError("fixed_params должен быть dict")
    keys = set(fixed_params)
    allowed = FIXED_KEYS | RND_KEYS
    extra = sorted(keys - allowed)
    missing = sorted(FIXED_KEYS - keys)
    if extra or missing:
        parts: list[str] = []
        if extra:
            parts.append(f"лишние {extra}")
        if missing:
            parts.append(f"отсутствуют {missing}")
        raise ValueError(f"fixed_params: {', '.join(parts)}")

    checked: dict[str, int | float | bool] = {
        key: _require_positive_int("fixed_params", key, fixed_params[key])
        for key in FIXED_KEYS - {"reward_delta"}
    }
    checked["reward_delta"] = _require_delta(fixed_params["reward_delta"])
    if checked["n_success"] > checked["success_window"]:
        raise ValueError("n_success не может быть больше success_window")

    # RND-параметры: дефолты + валидация.
    for key, default in RND_DEFAULTS.items():
        checked[key] = fixed_params.get(key, default)
    if type(checked["use_rnd"]) is not bool:
        raise TypeError("fixed_params['use_rnd'] должен быть bool")
    for key in ("rnd_feature_dim", "rnd_update_every"):
        value = checked[key]
        if type(value) is not int or value < 1:
            raise TypeError(f"fixed_params[{key!r}] должен быть целым >= 1")
    for key in ("rnd_lr", "rnd_beta", "rnd_beta_decay", "rnd_beta_min"):
        value = checked[key]
        if type(value) not in (int, float) or isinstance(value, bool) or value < 0:
            raise TypeError(f"fixed_params[{key!r}] должен быть числом >= 0")
        checked[key] = float(value)
    if not (0.0 < float(checked["rnd_beta_decay"]) <= 1.0):
        raise ValueError("rnd_beta_decay должен быть в (0, 1]")
    if float(checked["rnd_beta_min"]) > float(checked["rnd_beta"]):
        raise ValueError("rnd_beta_min не может быть больше rnd_beta")
    return checked


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
    fixed_params: dict[str, int | float | bool],
    changable_params: dict[str, str],
    seed: int,
) -> tuple[float, int, int]:
    """Обучает PPO на одной карте и пишет логи в ``run_dir/map_name``.

    :param map_name: полный id карты, например ``MiniGrid-Empty-8x8-v0``
    :param run_dir: папка запуска, имя вида ``laptop_cuda_ver0``
    :param fixed_params: лимиты обучения, окно успеха, ``reward_delta``
        и опциональные ключи RND (см. ``RND_KEYS``)
    :param changable_params: ``device`` — ``cpu`` или ``cuda``
    :param seed: общее зерно для всех карт и машин
    :return: время до остановки в секундах и число целей среди последних эпизодов окна
    """
    map_name = _require_map_name(map_name)
    run_dir = _require_run_dir(run_dir)
    fixed = _require_fixed(fixed_params)
    device = _require_device(changable_params, run_dir)
    seed = _require_seed(seed)

    use_rnd = bool(fixed["use_rnd"])

    map_dir = Path(run_dir) / map_name
    map_dir.mkdir(parents=True, exist_ok=True)
    stats = EpisodeStats(map_dir, fixed)
    fps_writer = FpsWriter(map_dir / "fps.txt")
    env: gym.Env | None = None
    model: PPO | None = None
    rnd_module: RNDModule | None = None
    try:
        env = ImgObsWrapper(gym.make(map_name, max_steps=fixed["max_episode_steps"]))
        if use_rnd:
            rnd_module = RNDModule(
                obs_shape=env.observation_space.shape,
                feature_dim=int(fixed["rnd_feature_dim"]),
                lr=float(fixed["rnd_lr"]),
                device=device,
            )
            env = IntrinsicRewardWrapper(
                env,
                rnd_module=rnd_module,
                beta=float(fixed["rnd_beta"]),
                beta_decay=float(fixed["rnd_beta_decay"]),
                beta_min=float(fixed["rnd_beta_min"]),
                update_every=int(fixed["rnd_update_every"]),
            )
        model = _build_model(env, device, seed)
        model.set_logger(Logger(folder=str(map_dir), output_formats=[fps_writer]))
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.learn(total_timesteps=fixed["total_timesteps"], callback=stats)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return elapsed, *stats.goal_count()
    finally:
        stats.close()
        if model is not None:
            model.logger.close()
        else:
            fps_writer.close()
        if rnd_module is not None:
            try:
                rnd_module.save(map_dir / "rnd_module.pt")
            except Exception:  # noqa: BLE001 — чекпоинт не должен валить прогон
                logger.exception("не удалось сохранить RND-чекпоинт")
        if env is not None:
            env.close()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Обучение PPO (опционально + RND) на одной карте.")
    parser.add_argument("--map", required=True, help="полный id карты MiniGrid")
    parser.add_argument("--run-dir", required=True, help="например, laptop_cpu_ver0")
    parser.add_argument("--fixed-params", required=True, help="JSON-словарь параметров")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seconds, goals, counted = train(
        args.map,
        args.run_dir,
        json.loads(args.fixed_params),
        {"device": args.device},
        args.seed,
    )
    print(f"{seconds:.1f}s, goals {goals}/{counted}")
