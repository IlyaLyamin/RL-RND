"""Обход карт MiniGrid на текущей машине и вызов сборки отчёта"""

MACHINE = "desktop"
SEED = 42
DEVICES = ("cpu", "cuda")
FIXED_PARAMS = {
    "max_episode_steps": 768,
    "total_timesteps": 150_000,
    "no_improve_episodes": 500,
    "n_success": 7,
    "success_window": 10,
    "reward_delta": 0.02,
}
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
        "maps": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _train_map(run_dir: Path, device: str, map_name: str) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        return {"map": map_name, "seconds": None, "goals": None, "counted": None, "error": "PyTorch не видит видеокарту"}
    try:
        seconds, goals, counted = train(map_name, str(run_dir), FIXED_PARAMS, {"device": device}, SEED)
    except Exception as exc:
        return {
            "map": map_name,
            "seconds": None,
            "goals": None,
            "counted": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"map": map_name, "seconds": seconds, "goals": goals, "counted": counted, "error": None}


def _run_device(ver: int, device: str) -> None:
    run_dir = ROOT / f"{MACHINE}_{device}_ver{ver}"
    summary_path = run_dir / "summary.json"
    rows: list[dict[str, object]] = []
    _write_summary(summary_path, device, ver, rows)

    for map_name in MAPS:
        logger.info("%s %s: старт %s", MACHINE, device, map_name)
        row = _train_map(run_dir, device, map_name)
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


def main() -> None:
    ver = next_ver(ROOT, MACHINE)
    logger.info("%s ver%s, seed %s, устройства %s", MACHINE, ver, SEED, ", ".join(DEVICES))
    try:
        for device in DEVICES:
            _run_device(ver, device)
    finally:
        logger.info("Отчёт: %s", build_report(ROOT, MACHINE, ver))


if __name__ == "__main__":
    main()
