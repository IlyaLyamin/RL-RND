"""Сборка md-отчёта: снимок карты, затем награда, длина эпизода и fps"""

SUMMARY_NAME = "summary.json"
PLOTS_SUFFIX = "_plots"
DEVICES = ("cpu", "cuda")
COLORS = {"cpu": "#1f77b4", "cuda": "#d62728"}
CHARTS = (
    ("reward.txt", "reward", "эпизод", "награда", "Награда по эпизодам, без сглаживания."),
    ("steps_per_ep.txt", "steps", "эпизод", "действий за эпизод", "Длина эпизода в действиях."),
    ("fps.txt", "fps", "шаг", "fps", "Скорость обучения, кадров среды в секунду."),
)

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]


def latest_ver(root: Path, machine: str) -> int:
    """Самый большой номер ver среди папок этой машины."""
    pattern = re.compile(rf"^{re.escape(machine)}_(?:cpu|cuda)_ver(\d+)$")
    numbers = [
        int(match.group(1))
        for path in root.iterdir()
        if (match := pattern.fullmatch(path.name))
    ]
    if not numbers:
        raise FileNotFoundError(f"для машины {machine} нет папок запуска")
    return max(numbers)


def _read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(summary: dict | None) -> dict[str, dict]:
    if summary is None:
        return {}
    return {row["map"]: row for row in summary.get("maps", [])}


def _read_xy(path: Path) -> tuple[list[float], list[float]]:
    if not path.exists():
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        x, y = line.split()
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def _save_chart(path: Path, series: list[tuple[str, list[float], list[float]]], xlabel: str, ylabel: str) -> bool:
    drawn = [(device, xs, ys) for device, xs, ys in series if xs]
    if not drawn:
        return False
    figure, axes = plt.subplots(figsize=(8, 4), dpi=120)
    try:
        for device, xs, ys in drawn:
            axes.plot(xs, ys, label=device.upper(), color=COLORS[device], linewidth=1)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.grid(True, alpha=0.3)
        axes.legend()
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
    finally:
        plt.close(figure)
    return True


def _time_cell(row: dict | None) -> str:
    if row is None or type(row.get("seconds")) not in (int, float):
        return "—"
    return f"{float(row['seconds']):.1f}"


def _success_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    if row.get("error"):
        return "ошибка"
    if row.get("success") is True:
        return "да"
    if row.get("success") is False:
        return "нет"
    return "—"


def _charts(root: Path, machine: str, ver: int, map_name: str, plots_dir: Path) -> list[str]:
    lines: list[str] = []
    for filename, stem, xlabel, ylabel, caption in CHARTS:
        series = []
        for device in DEVICES:
            xs, ys = _read_xy(root / f"{machine}_{device}_ver{ver}" / map_name / filename)
            series.append((device, xs, ys))
        image = plots_dir / f"{map_name}_{stem}.png"
        if _save_chart(image, series, xlabel, ylabel):
            lines.extend([f"![{caption}]({image.relative_to(root).as_posix()})", "", caption, ""])
        else:
            lines.extend([f"{caption} Нет данных.", ""])
    return lines


def build_report(root: Path, machine: str, ver: int | None = None) -> Path:
    """Собирает md по указанной версии или по последней, если версия не задана."""
    if ver is None:
        ver = latest_ver(root, machine)

    summaries = {
        device: _read_summary(root / f"{machine}_{device}_ver{ver}" / SUMMARY_NAME)
        for device in DEVICES
    }
    by_device = {device: _rows(summary) for device, summary in summaries.items()}
    map_names: list[str] = []
    for device in DEVICES:
        for map_name in by_device[device]:
            if map_name not in map_names:
                map_names.append(map_name)

    summary = summaries.get("cpu") or summaries.get("cuda")
    lines = [f"# {machine} ver{ver}", ""]
    if summary is None:
        lines.extend(["Сводный файл запуска не найден.", ""])
    else:
        fixed = summary.get("fixed_params") or {}
        lines.extend([
            f"Машина: {machine}. Seed: {summary.get('seed')}. Версия: {ver}.",
            (
                f"Лимит шагов эпизода: {fixed.get('max_episode_steps', '—')}. "
                f"Бюджет шагов: {fixed.get('total_timesteps', '—')}. "
                f"Остановка без роста награды: {fixed.get('no_improve_episodes', '—')} эпизодов."
            ),
            (
                f"Успех: не меньше {summary.get('n_success', '—')} из последних "
                f"{summary.get('success_window', '—')} эпизодов. "
                "В таблице время до этой остановки, в секундах."
            ),
            "",
        ])

    lines.extend([
        "## Время и успех",
        "",
        "| Карта | CPU, с | CPU успех | CUDA, с | CUDA успех |",
        "| --- | ---: | --- | ---: | --- |",
    ])
    for map_name in map_names:
        cpu = by_device["cpu"].get(map_name)
        cuda = by_device["cuda"].get(map_name)
        lines.append(
            f"| {map_name} | {_time_cell(cpu)} | {_success_cell(cpu)} | "
            f"{_time_cell(cuda)} | {_success_cell(cuda)} |"
        )
    lines.append("")

    plots_dir = root / f"{machine}_ver{ver}{PLOTS_SUFFIX}"
    lines.extend(["## Карты", ""])
    for map_name in map_names:
        lines.extend([f"### {map_name}", ""])
        photo = root / "maps" / f"{map_name}.png"
        if photo.exists():
            lines.extend([f"![{map_name}]({photo.relative_to(root).as_posix()})", ""])
        else:
            lines.extend(["Снимок карты не найден.", ""])
        lines.extend(_charts(root, machine, ver, map_name, plots_dir))

    report_path = root / f"{machine}_ver{ver}.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    print(build_report(Path(__file__).resolve().parent, "desktop"))
