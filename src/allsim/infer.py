import logging
from abc import abstractmethod
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


class Condition:
    def __init__(self, labels: np.ndarray, function: Callable, options: int):
        self.labels = labels
        self.function = function
        self.options = options

    def __len__(self):
        return self.options

    def __call__(self, row: pd.Series) -> int:
        return self.function(row[self.labels])


class Density:
    def __init__(
        self,
        condition: Condition,
        K: int = 1,
        drop: np.ndarray = np.array(["condition"]),
    ):
        super().__init__()

        self.condition = condition
        self.shape = (0, 0)
        self.columns = []
        self.drop = np.unique(np.array([*drop, "condition"]))

        if condition is not None:
            if K != len(condition):
                logging.warn(
                    f"Given K does not match len(condition), set K to {len(condition)}"
                )
                K = len(condition)
        self.K = K

    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        ...

    @abstractmethod
    def fit(self, D: pd.DataFrame) -> None:
        self.shape = D.shape
        self.columns = D.columns.values


class KDEDensity(Density):
    def __init__(
        self,
        condition: Condition = None,
        K: int = 1,
        cluster_kwargs: dict = {},
        drop: np.ndarray = np.array(["condition"]),
    ):
        super().__init__(condition=condition, K=K, drop=drop)

        self.densities = dict()

    def fit(
        self,
        D: pd.DataFrame,
        one_hot_encoded: np.ndarray = None,
        cluster_kwargs: dict = {},
    ) -> None:
        super().fit(D)

        self.D = D.copy(deep=True)
        self.one_hot_encoded = one_hot_encoded

        self.D = self._set_condition_on_data(self.D, cluster_kwargs=cluster_kwargs)

        for condition in self.D.condition.unique():
            data = self.D[self.D.condition == condition]

            params = {"bandwidth": np.logspace(-1, 1, 5)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(data.drop(self.drop, axis=1))

            self.densities[condition] = grid.best_estimator_

    def sample(self, n: int = 1, condition: np.ndarray = None) -> np.ndarray:
        if condition is not None:
            assert n == len(
                condition
            ), f"n must match amount of conditions, but n ({n}) != len(condition) ({len(condition)})."
        else:
            condition = np.random.choice(self.D.condition.unique(), size=n)

        columns = self.D.drop(self.drop, axis=1, errors="ignore").columns.values
        samples = pd.DataFrame(columns=columns)

        samples_data = np.array([self.densities[c].sample() for c in condition])
        if len(samples_data) > 0:
            samples = pd.DataFrame(columns=columns, data=samples_data.reshape(n, -1))
            samples = self._cleanup_samples(samples)

        return samples

    def _cleanup_samples(self, samples: pd.DataFrame) -> np.ndarray:
        if self.one_hot_encoded is not None:
            for group in self.one_hot_encoded:
                group_values = samples[group].to_numpy()
                code = np.zeros_like(group_values)
                code[np.arange(len(group_values)), group_values.argmax(1)] = 1
                samples.loc[:, group] = code
        return samples

    def _fit_cluster(self, D: pd.DataFrame, K: int = 8, **kwargs) -> pd.DataFrame:
        kmeans = KMeans(n_clusters=K, **kwargs).fit(D)
        self.kmeans = kmeans

        self.D.loc[:, "condition"] = kmeans.predict(
            self.D.drop(self.drop, errors="ignore")
        )

        return self.D

    def _set_condition_on_data(
        self, D: pd.DataFrame, cluster_kwargs: dict = {}
    ) -> pd.DataFrame:
        if self.condition is None:
            D = self._fit_cluster(D, K=self.K, **cluster_kwargs)
        else:
            D.loc[:, "condition"] = D.apply(self.condition, axis=1)

        return D


class ArrivalProcess:
    def __init__(self):
        self.current_time = 0
        self.current_value = 0

    def __call__(self, t: int, **kwargs) -> int:
        self.current_time += 1
        self.current_value = self.progress(self.current_time)

        return self.progress(t, **kwargs)

    @abstractmethod
    def progress(self, t: int, **kwargs) -> int:
        ...


class System:
    def __init__(self, density: Density, system: Dict[int, ArrivalProcess]):
        assert density.K == len(
            system
        ), f"K of density should match amount of processes in system, found {len(system)} but require {density.K}."

        self.system = system
        self.density = density

    @abstractmethod
    def __call__(self, t: float) -> pd.DataFrame:
        assert t > 0, f"t cannot be {t}; must be larger than 0 (t > 0)."


class PoissonProcess(ArrivalProcess):
    def __init__(
        self, lam: float = 1, update_lam: Callable[[int], float] = lambda t: t
    ):
        super().__init__()

        self._baseline_lam = lam
        self.lam = lam
        self.update_lam = update_lam

    def get_lam_unnormalized(self, t: int) -> float:
        return self._baseline_lam * self.update_lam(t)

    def progress(self, t: int, neu: float = 1) -> int:
        self.lam = neu * self.get_lam_unnormalized(t)  # eqs. (5, 6)
        return np.random.poisson(lam=self.lam)


class PoissonSystem(System):
    def __init__(
        self,
        density: Density,
        system: Dict[int, PoissonProcess],
        normalize: bool = True,
        alpha: Callable[[int], float] = lambda t: 1,
    ):
        super().__init__(density=density, system=system)

        self.normalize, self.alpha = normalize, alpha

    def __call__(self, t: float) -> pd.DataFrame:
        super().__call__(t=t)

        # 1. for each condition -> ArrivalProcess, calculate the Poisson arrival rate
        arrival_rates = {
            c: process.get_lam_unnormalized(t) for c, process in self.system.items()
        }

        # 2. if normalize is set True, calculate neu
        if self.normalize:
            if np.array(list(arrival_rates.values())).sum():
                neu = self.alpha(t) / np.array(list(arrival_rates.values())).sum()
            else:
                neu = 0
        else:
            neu = 1
            self.alpha = lambda t: np.array(list(arrival_rates.values())).sum()

        # 3. sample amount of objects
        arrival_rates = {c: process(t, neu=neu) for c, process in self.system.items()}

        # 4. sample objects, per amount of condition
        condition = np.repeat(
            list(arrival_rates.keys()), repeats=list(arrival_rates.values())
        )
        return self.density.sample(n=len(condition), condition=condition)
