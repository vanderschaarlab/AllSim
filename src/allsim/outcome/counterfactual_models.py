from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans
from torch import nn, optim
from torchmetrics import MeanSquaredError

from .utils import GradientReversal


class OrganITE_Network(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        gamma: float,
        lambd: float,
        kappa: float,
        weight_decay: float,
        n_clusters: int,
        num_hidden_layers: int = 1,
        activation_type: str = "relu",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.gamma = gamma
        self.lambd = lambd
        self.kappa = kappa
        self.weight_decay = weight_decay

        self.n_clusters = n_clusters

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # phi

        activation_functions = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}
        activation = activation_functions[activation_type]

        hidden_layers = np.array(
            [
                (
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    activation(),
                    nn.Dropout(p=dropout_prob),
                )
                for _ in range(num_hidden_layers)
            ]
        ).flatten()

        # NETWORK
        self.representation = nn.Sequential(  # This predicts phi
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),
            *hidden_layers,
            nn.Linear(self.hidden_dim, self.output_dim),
            activation(),
        )

        self.propensity = nn.Sequential(  # p(c(O) | Phi)
            GradientReversal(self.lambd),
            nn.Linear(output_dim, output_dim),
            activation(),
            nn.Linear(output_dim, self.n_clusters),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(  # Y | Phi
            nn.Linear(output_dim, output_dim), activation(), nn.Linear(output_dim, 1)
        )

        self.loss_mse = MeanSquaredError()
        self.loss_cel = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    # ~~~~~~~~~~~~~
    # TORCH METHODS

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.representation(x)
        p = self.propensity(phi)
        y = self.output(phi)

        return p, y

    # ~~~~~~~~~~~
    # LIT METHODS

    def on_fit_start(self, stage: Any = None) -> None:
        self.cluster = KMeans(n_clusters=self.n_clusters)
        _, O, _, _ = self.trainer.datamodule.train_dataloader().dataset.dataset.tensors

        self.cluster.fit(O)

        self.x_cols = self.trainer.datamodule.x_cols
        self.o_cols = self.trainer.datamodule.o_cols

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["cluster"] = self.cluster

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.cluster = checkpoint["cluster"]

    def configure_optimizers(self) -> tuple:
        optimiser = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, self.gamma)

        return [optimiser], [scheduler]

    # INFERENCE

    def shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, o, y, _ = batch

        u = torch.cat((x, o), dim=1)
        p, y_ = self.forward(u)

        c = self.cluster.predict(o.cpu())
        c = torch.Tensor(c).to(self.device).long()

        mse = self.loss_mse(y_, y)
        prop = self.loss_cel(p, c)

        return mse, prop  # scale prop

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ix: int,
    ) -> torch.Tensor:
        mse, prop = self.shared_step(batch)
        loss = mse + self.kappa * prop

        self.log("train_loss - MSE", mse, on_epoch=True)
        self.log("train_loss - Prop.", prop, on_epoch=True)
        self.log("train_loss - total", loss, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ix: int,
    ) -> torch.Tensor:
        mse, prop = self.shared_step(batch)
        loss = mse + self.kappa * prop

        self.log("val_loss - MSE", mse, on_epoch=True)
        self.log("val_loss - Prop.", prop, on_epoch=True)
        self.log("val_loss - total", loss, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ix: int,
    ) -> torch.Tensor:
        x, o, y, _ = batch

        _, y_ = self.forward(torch.cat((x, o), dim=1))

        rmse = torch.sqrt(self.loss_mse(y_, y))

        # SCALE
        mean, std = self.trainer.datamodule.mean, self.trainer.datamodule.std
        y = y * std + mean
        y_ = y_ * std + mean

        loss = torch.abs(y - y_)
        self.log("test_loss - mean difference in days", loss, on_epoch=True)
        self.log("test_loss - RMSE", rmse, on_epoch=True)

        return loss
