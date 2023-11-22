import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import torch

from allsim.outcome.counterfactual_models import OrganITE_Network


class Inference(ABC):
    def __init__(self, model: Any, mean: float, std: float) -> None:
        assert isinstance(mean, (float, int)), "mean must be float or int"
        assert isinstance(std, (float, int)), "std must be float or int"
        self.model = model.eval()
        self.mean = mean
        self.std = std

    def __call__(
        self, x: torch.Tensor, r: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        return self.infer(x, r, *args, **kwargs)

    @abstractmethod
    def infer(self, x: torch.Tensor, r: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        ...


class Inference_OrganITE(Inference):
    def __init__(
        self,
        model: OrganITE_Network,
        mean: float,
        std: float,
        x_indices_real: np.ndarray = None,
        r_indices_real: np.ndarray = None,
        x_mean: np.ndarray = None,
        x_scale: np.ndarray = None,
        r_mean: np.ndarray = None,
        r_scale: np.ndarray = None,
    ) -> None:
        super().__init__(model, mean, std)

        self.x_indices_real = x_indices_real
        self.r_indices_real = r_indices_real
        self.x_mean, self.x_scale = x_mean, x_scale
        self.r_mean, self.r_scale = r_mean, r_scale

    def _inverse_transform(self, x, r) -> Tuple[np.ndarray, np.ndarray]:
        if (
            x is None
            or self.x_indices_real is None
            or self.x_scale is None
            or self.x_mean is None
        ):
            x = x
        else:
            x[:, self.x_indices_real] = (
                x[:, self.x_indices_real] * self.x_scale + self.x_mean
            )

        if (
            r is None
            or self.r_indices_real is None
            or self.r_scale is None
            or self.x_mean is None
        ):
            r = r
        else:
            r[:, self.r_indices_real] = (
                r[:, self.r_indices_real] * self.r_scale + self.r_mean
            )

        return x, r

    def _transform(
        self, x: np.ndarray, r: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (
            x is None
            or self.x_indices_real is None
            or self.x_scale is None
            or self.x_mean is None
        ):
            x = x
        else:
            x[:, self.x_indices_real] = (
                x[:, self.x_indices_real] - self.x_mean
            ) / self.x_scale

        if (
            r is None
            or self.r_indices_real is None
            or self.r_scale is None
            or self.x_mean is None
        ):
            r = r
        else:
            r[:, self.r_indices_real] = (
                r[:, self.r_indices_real] - self.r_mean
            ) / self.r_scale

        return x, r

    def infer(
        self, x: torch.Tensor, r: Optional[torch.Tensor] = None, replace_organ: int = -1
    ) -> Any:  # type: ignore
        # NOTE: replace_organ should fit what has been defined
        #   at training time

        x = copy.deepcopy(x)
        r = copy.deepcopy(r)

        x, r = self._transform(x, r)

        with torch.no_grad():
            if r is None:
                r = np.full((len(x), len(self.model.o_cols)), replace_organ)
            x = torch.Tensor(x).double()
            r = torch.Tensor(r).double()
            catted = torch.cat((x, r), dim=1)

            _, y = self.model(catted)
            y = y * self.std + self.mean

            return np.abs(y.numpy())
