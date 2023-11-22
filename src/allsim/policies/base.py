from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Patient:
    covariates: np.ndarray
    time_to_live: int
    id: int

    def age(self, days: int) -> "Patient":
        self.time_to_live -= days
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Patient):
            return NotImplemented
        return self.id == other.id

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Patient):
            return NotImplemented
        return self.id > other.id


@dataclass
class Stats:
    deaths: int = 0
    patients_seen: int = 0
    population_life_years: float = 0
    transplant_count: int = 0
    first_empty_day: int = -1
    patients_transplanted: dict = field(
        default_factory=dict
    )  # should be a dict of day: np.array w shape (2, n_patients_on_day)
    organs_transplanted: dict = field(default_factory=dict)
    patients_died: dict = field(default_factory=dict)  # dict of day: np.array

    def __str__(self) -> str:
        return f"Deaths: {self.deaths}\nPopulation life-years: {self.population_life_years}\nTransplant count: {self.transplant_count}\nFirst empty day: {self.first_empty_day}"
