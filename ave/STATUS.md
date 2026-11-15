# RND на MiniGrid — статус, что починить, как запускать

Дата: финал недели 1 из 4. Ветка: Путь B (одна value-голова, RND во wrapper'е).
Рабочая папка: RL-RND/ave/ . venv активировать из RL-RND/venv/ .

---

## 0. TL;DR — что сделать в первую очередь

1. Применить ФИКС №1 (RunningMeanStd.normalize, 2 строки).
2. `pytest tests/test_rnd.py -q` → должно быть `4 passed`.
3. Перезапустить RND-прогоны на 3 картах (smoke 50k), проверить `reward.txt` эпизод 1 = `~1–10`.
4. Снизить бюджет в `collect.py` до `200_000` / `500`.
5. `python collect.py` и `python collect.py --rnd`.
6. `python report.py` → открыть `desktop_verN.md`.
7. Написать `README_RND.md` (черновик ниже, раздел 6).

---

## 1. Что уже работает (не трогать)

- Baseline PPO (`use_rnd=False`) — идентичен исходному `train.py`. Smoke: `(107s, 100, 100)` на MultiRoom, `(97s, 0, 100)` на KeyCorridor, `(101s, 7, 100)` на DoorKey.
- RND (`use_rnd=True, rnd_beta=0.1, rnd_beta_decay=1.0`):
  - MultiRoom-N2-S4: `100/100` (как baseline).
  - **KeyCorridorS3R3: `100/100` против `0/100` у baseline — главный результат.**
  - DoorKey-5x5: `100/100` против `7/100` у baseline.
- `rnd_loss` падает в ~1800–3000 раз за 50k шагов — predictor учится.
- `best_model` выбирается по `r_ext` (фикс шага 2 применён) — не залипает на novelty-выбросе.
- `intrinsic_reward.txt` пишется, формат: `эпизод Σ r_int_scaled средний_rnd_loss beta`.

---

## 2. ЧТО ПОЧИНИТЬ (по приоритету)

### ФИКС №1 (обязательный, 2 строки) — `RunningMeanStd.normalize`

**Файл:** `src/rnd/running_mean_std.py`

**Проблема:** на первых шагах `reward_rms.var ≈ 0`, деление `err / sqrt(var + 1e-8)` даёт выброс до `5` (после клиппинга). Из-за этого `reward.txt` эпизод 1 = `57` вместо `~1–10`, а `best_model` обновляется позже, чем мог бы. Успех `100/100` это **не сломало**, но в отчёте будет неприятная оговорка и RND выглядит медленнее, чем есть.

**Заменить метод `normalize` на:**

```python
def normalize(self, x: np.ndarray) -> np.ndarray:
    """``(x - mean) / sqrt(var + eps)``. Не меняет статистики.
    Пока не набрано >= 2 наблюдений, возвращает x как есть:
    иначе деление на sqrt(0 + 1e-8) даёт выбросы на первых шагах."""
    arr = np.asarray(x, dtype=np.float64)
    if self.count < 2:
        return arr
    return (arr - self.mean) / np.sqrt(self.var + self._eps)
```

**После фикса:**

- `pytest tests/test_rnd.py -q` → `4 passed`.
- Перезапустить RND-прогоны на 3 картах (команды в разделе 4), проверить `Get-Content .../reward.txt -TotalCount 3` — эпизод 1 должен быть `~1–10`, не `50+`.

### ФИКС №2 (косметика) — сообщение об остановке

**Файл:** `train.py`, метод `EpisodeStats._on_step`.

**Проблема:** в логе пишется `Лучшая награда 0.977500`, хотя для RND это `r_ext`, а не суммарный `reward`. Технически верно, но читается как баг.

**Опционально:** заменить строку лога на

```python
logger.info(
    "Остановка: метрика не выросла больше чем на %s за %s эпизодов. Лучшая метрика %.6f%s",
    self._reward_delta,
    self._no_improve_episodes,
    self._best_reward,
    " (r_ext)" if self._use_rnd else "",
)
```

Не блокер, можно оставить как есть.

### ФИКС №3 (важный для финального прогона) — бюджет в `collect.py`

**Файл:** `collect.py`, `FIXED_PARAMS`.

**Проблема:** сейчас `total_timesteps=1_000_000`, `no_improve_episodes=500`. На `KeyCorridorS3R3` baseline за 1M шагов, скорее всего, всё равно не сойдётся (`0/100` при 50k), а прогон займёт часы. Для курсовой хватает `200k`.

**Заменить на:**

```python
FIXED_PARAMS = {
    "max_episode_steps": 768,
    "total_timesteps": 200_000,
    "no_improve_episodes": 500,
    "n_success": 7,
    "success_window": 10,
    "reward_delta": 0.02,
}
```

На CPU 3 карты × 200k × (baseline + RND) ≈ 1–2 часа.

### ФИКС №4 (опциональный) — `intrinsic_reward.txt` в отчёте

**Файл:** `report.py`, `CHARTS`.

**Проблема:** `report.py` не строит график `r_int`/`rnd_loss`. Формат `intrinsic_reward.txt` — 4 колонки (`эпизод Σ r_int_scaled rnd_loss beta`), `_read_xy` ждёт 2. Если добавить в `CHARTS`, `_read_xy` упадёт на `line.split()` (`too many values to unpack`).

**Не блокер.** В отчёт `intrinsic_reward.txt` вставить таблицей/скриншотом. График — на неделю 3–4, если будет время (нужен отдельный ридер для 4-колоночного формата).

---

## 3. Что НЕ трогать (ограничения проекта)

- Не делать кастомную policy с двумя value-головами — это Путь A, неделя 3–4.
- Не трогать 17 карт из `MAPS` — только `ACTIVE_MAPS` (3 карты).
- Не менять гиперпараметры PPO (lr, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef).
- Не добавлять reward shaping (reward за приближение к цели и т.п.) — baseline должен быть «чистым».
- Не реализовывать «предобучить на простой → кинуть на сложную» — уходит в «ограничения» отчёта.
- Не переносить `venv/` — сломается (абсолютные пути). Активировать из `RL-RND/venv/`.

---

## 4. Как запускать (PowerShell, Windows)

### 4.1. Активация venv и переход в рабочую папку

```powershell
D:\ML_vk\RL-RND\venv\Scripts\Activate.ps1
cd D:\ML_vk\RL-RND\ave
```

Все команды ниже — **из `ave/`**. Иначе `from src.rnd import ...` упадёт с `ModuleNotFoundError`.

### 4.2. Тесты

```powershell
pytest tests/test_rnd.py -q
```

Ожидаемо: `4 passed`.

### 4.3. Smoke baseline (50k, ~1.5 мин)

```powershell
python -c "from train import train; print(train('MiniGrid-MultiRoom-N2-S4-v0','laptop_cpu_ver0',{'max_episode_steps':200,'total_timesteps':50000,'no_improve_episodes':1000,'n_success':5,'success_window':100,'reward_delta':0.01,'use_rnd':False},{'device':'cpu'},42))"
```

Ожидаемо: `(~107, 100, 100)`.

### 4.4. Smoke RND (50k, ~2–3 мин)

```powershell
python -c "from train import train; print(train('MiniGrid-MultiRoom-N2-S4-v0','laptop_cpu_ver1',{'max_episode_steps':200,'total_timesteps':50000,'no_improve_episodes':1000,'n_success':5,'success_window':100,'reward_delta':0.01,'use_rnd':True,'rnd_beta':0.1,'rnd_beta_decay':1.0},{'device':'cpu'},42))"
```

Ожидаемо: `(~180, 100, 100)`, в папке `laptop_cpu_ver1/MiniGrid-MultiRoom-N2-S4-v0/` есть `intrinsic_reward.txt` и `rnd_module.pt`.

### 4.5. Smoke RND на сложной карте (главный результат)

```powershell
python -c "from train import train; print(train('MiniGrid-KeyCorridorS3R3-v0','laptop_cpu_ver1',{'max_episode_steps':200,'total_timesteps':50000,'no_improve_episodes':1000,'n_success':5,'success_window':100,'reward_delta':0.01,'use_rnd':True,'rnd_beta':0.1,'rnd_beta_decay':1.0},{'device':'cpu'},42))"
```

Ожидаемо: `(~198, 100, 100)`.

### 4.6. Полный прогон через `collect.py`

```powershell
python collect.py          # baseline, 3 карты × cpu/cuda, новый ver
python collect.py --rnd    # RND, 3 карты × cpu/cuda, следующий ver
```

После каждого — строка `Отчёт: ...\desktop_verN.md`.

На машине без GPU CUDA-часть пометится ошибкой и пропустится; CPU-часть отработает.

### 4.7. Сборка отчёта

```powershell
python report.py
```

Открыть `desktop_verN.md` в VS Code preview (Ctrl+Shift+V).

### 4.8. PowerShell: JSON с одинарными кавычками ломается

`--fixed-params '{...}'` через PowerShell **не работает** (`JSONDecodeError: Expecting property name enclosed in double quotes`). PowerShell срезает одинарные кавычки перед передачей в `python.exe`.

**Обходной путь:** `python -c "from train import train; ..."` (как выше). Либо экранировать двойные кавычки бэктиком: `'{\"max_episode_steps\":200,...}'` — но `-c` надёжнее.

`collect.py` это не касается — у него флаги `--rnd`/`--all-maps`, без `--fixed-params`.

---

## 5. Где лежат результаты и логи

``` text
ave/
├── laptop_cpu_ver0/                    # baseline-прогоны (smoke)
│   └── MiniGrid-<Map>-v0/
│       ├── reward.txt                  # эпизод, суммарная награда
│       ├── steps_per_ep.txt            # эпизод, шагов
│       ├── fps.txt                     # шаг, fps
│       ├── best_model.zip              # модель на лучшем эпизоде
│       └── (intrinsic_reward.txt нет)
├── laptop_cpu_ver1/                    # RND-прогоны (smoke)
│   └── MiniGrid-<Map>-v0/
│       ├── reward.txt
│       ├── steps_per_ep.txt
│       ├── fps.txt
│       ├── best_model.zip              # по r_ext
│       ├── intrinsic_reward.txt        # эпизод Σ r_int_scaled rnd_loss beta
│       └── rnd_module.pt               # чекпоинт RND (target+predictor+optimizer+obs_rms+reward_rms)
├── desktop_cpu_verN/                   # полный прогон через collect.py
│   └── summary.json                    # сводка по картам
├── desktop_verN.md                     # отчёт (из report.py)
└── maps/                               # снимки карт (для отчёта)
```
