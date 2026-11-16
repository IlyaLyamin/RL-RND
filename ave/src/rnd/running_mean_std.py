"""Потоковая нормализация наблюдений и награды (Welford)."""

import numpy as np


class RunningMeanStd:
    """Считает среднее и дисперсию онлайн по Уэлфорду.

    :param shape: форма входа; ``()`` — для скаляров (нормализация r_int)
    """

    def __init__(self, shape: tuple[int, ...] = (), eps: float = 1e-8) -> None:
        if type(shape) is not tuple:
            raise TypeError("shape должен быть tuple")
        self._eps = float(eps)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0.0

    def update(self, x: np.ndarray) -> None:
        """Обновляет статистики по батчу ``x`` формы ``(N, *shape)`` или ``(*shape,)``."""
        arr = np.asarray(x, dtype=np.float64)
        # Приводим к (N, *shape), чтобы одинаково обрабатывать одиночный obs и батч.
        if arr.shape == self.mean.shape:
            arr = arr[None, ...]
        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        self._merge(batch_mean, batch_var, float(batch_count))

    def _merge(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        """Параллельное слияние двух наборов статистик (Chan et al.)."""
        if batch_count <= 0:
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """``(x - mean) / sqrt(var + eps)``. Не меняет статистики.
        Если статистик ещё нет (count=0), возвращает x как есть."""
        arr = np.asarray(x, dtype=np.float64)
        if self.count < 2:
            return arr
        return (arr - self.mean) / np.sqrt(self.var + self._eps)

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
