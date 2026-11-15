"""Обход карт MiniGrid на текущей машине и вызов сборки отчёта.

Путь B: baseline и RND запускаются как два последовательных вызова
``main()`` (с разными ``use_rnd``) — каждому достанется свой ``ver``.
Карты основного эксперимента — ``ACTIVE_MAPS``. Полный ``MAPS``
оставлен для истории/расширения.
"""

MACHINE = "desktop"
SEED = 42
DEVICES = ("cpu", "cuda")
# Параметры baseline. Для RND-прогона main(run_rnd=True) добавит RND-ключи.
FIXED_PARAMS = {
    "max_episode_steps": 768,
    "total_timesteps": 1_000_000,
    "no_improve_episodes": 500,
    "n_success": 7,
    "success_window": 10,
    "reward_delta": 0.02,
}
# RND-параметры, добавляемые поверх FIXED_PARAMS при run_rnd=True.
RND_PARAMS = {
    "use_rnd": True,
    "rnd_feature_dim": 128,
    "rnd_lr": 1e-4,
    "rnd_beta": 0.1,
    "rnd_beta_decay": 1.0,
    "rnd_beta_min": 0.0,
    "rnd_update_every": 1,
}
# Карты основного эксперимента. Остальные 17 — в MAPS для истории.
ACTIVE_MAPS = (
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-DoorKey-5x5-v0",
)
# Полный список карт (не запускается в основном прогоне).
MAPS = (
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N6-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-LavaGapS7-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-DistShift1-v0",
    "MiniGrid-Dynamic-Obstacles-6x6-v0",
    "MiniGrid-Dynamic-Obstacles-16x16-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-16x16-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-ObstructedMaze-1Dlh-v0",
    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-MemoryS7-v0",
)

import json
import logging
import re
from pathlib import Path

import torch

from report import build_report
from train import train

ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("collect")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


def next_ver(root: Path, machine: str) -> int:
    """Следующий номер версии для папок cpu и cuda этой машины."""
    pattern = re.compile(rf"^{re.escape(machine)}_(?:cpu|cuda)_ver(\d+)$")
    numbers = [
        int(match.group(1))
        for path in root.iterdir()
        if (match := pattern.fullmatch(path.name))
    ]
    return max(numbers, default=-1) + 1


def _write_summary(path: Path, device: str, ver: int, rows: list[dict[str, object]]) -> None:
    payload = {
        "machine": MACHINE,
        "device": device,
        "ver": ver,
        "seed": SEED,
        "fixed_params": FIXED_PARAMS,
        "rnd_params": RND_PARAMS,
        "maps": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _train_map(
    run_dir: Path,
    device: str,
    map_name: str,
    params: dict[str, object],
) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        return {
            "map": map_name,
            "seconds": None,
            "goals": None,
            "counted": None,
            "error": "PyTorch не видит видеокарту",
        }
    try:
        seconds, goals, counted = train(
            map_name, str(run_dir), params, {"device": device}, SEED
        )
    except Exception as exc:  # noqa: BLE001 — прогон не должен падать целиком
        return {
            "map": map_name,
            "seconds": None,
            "goals": None,
            "counted": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"map": map_name, "seconds": seconds, "goals": goals, "counted": counted, "error": None}


def _run_device(ver: int, device: str, maps: tuple[str, ...], params: dict[str, object]) -> None:
    run_dir = ROOT / f"{MACHINE}_{device}_ver{ver}"
    summary_path = run_dir / "summary.json"
    rows: list[dict[str, object]] = []
    _write_summary(summary_path, device, ver, rows)

    for map_name in maps:
        logger.info("%s %s: старт %s", MACHINE, device, map_name)
        row = _train_map(run_dir, device, map_name, params)
        rows.append(row)
        _write_summary(summary_path, device, ver, rows)
        if row["error"]:
            logger.info("%s %s %s: %s", MACHINE, device, map_name, row["error"])
        else:
            logger.info(
                "%s %s %s: %.1f с, цели %s из %s",
                MACHINE,
                device,
                map_name,
                row["seconds"],
                row["goals"],
                row["counted"],
            )


def main(*, run_rnd: bool = False, maps: tuple[str, ...] = ACTIVE_MAPS) -> None:
    """Один прогон по ``maps``. ``run_rnd`` включает RND-обёртку.

    Каждый вызов ``main`` получает следующий ``ver`` (baseline → ver0,
    RND → ver1 и т.д.). Отчёт собирается в ``finally``.
    """
    params: dict[str, object] = dict(FIXED_PARAMS)
    if run_rnd:
        params.update(RND_PARAMS)
    else:
        params["use_rnd"] = False

    ver = next_ver(ROOT, MACHINE)
    logger.info(
        "%s ver%s, seed %s, устройства %s, RND=%s, карт %s",
        MACHINE,
        ver,
        SEED,
        ", ".join(DEVICES),
        run_rnd,
        len(maps),
    )
    try:
        for device in DEVICES:
            _run_device(ver, device, maps, params)
    finally:
        logger.info("Отчёт: %s", build_report(ROOT, MACHINE, ver))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обход карт MiniGrid и сборка отчёта.")
    parser.add_argument(
        "--rnd",
        action="store_true",
        help="включить RND (иначе baseline)",
    )
    parser.add_argument(
        "--all-maps",
        action="store_true",
        help="прогнать полный MAPS вместо ACTIVE_MAPS",
    )
    args = parser.parse_args()
    main(run_rnd=args.rnd, maps=MAPS if args.all_maps else ACTIVE_MAPS)
