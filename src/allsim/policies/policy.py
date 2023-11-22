import copy
from abc import ABC, abstractclassmethod
from operator import attrgetter
from typing import Any, Tuple

import numpy as np
import pandas as pd
import scipy
import torch

from allsim.data.data_module import OrganDataModule
from allsim.outcome.counterfactual_inference import Inference
from allsim.policies.base import Patient

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# POLICY DEFINITIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# LivSim Policies (MELD and MELD-na)


class Policy(ABC):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information
        data: str = "test",
        #   of the transplant system
    ) -> None:
        self.name = name
        self.waitlist = copy.deepcopy(initial_waitlist)
        self.dm = dm
        if data == "test":
            self.test = dm._test_processed  # perform on test set
        if data == "all":
            self.test = dm._all_processed  # perform on all data

    @abstractclassmethod
    def select(self, organs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Given the internal state of the transplant system
        # waitlist, and a new organ, a patient is suggested.
        # For each patient the ID is used/returned; the policy may
        # use dm for full covariates. When the patient is presented
        # they should be removed from the waitlist.
        #
        # params -
        # :organ: int - organ ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        ...

    @abstractclassmethod
    def add(self, x: np.ndarray) -> None:
        # Whenever a patient enters the transplant system
        # add_x is called. It is the policies task to maintain
        # system state.
        #
        # params -
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        ...

    @abstractclassmethod
    def remove(self, x: np.ndarray) -> None:
        # Removes x from the waitlist; happens when they
        # died prematurely. It is the Sim's responsibility
        # to define when a patients dies. As long as the patient
        # remains on the waitlist, they are assumed to be alive.
        #
        # params -
        # :x: int - patient ID, for reference to dm.ID (note,
        #   dm.ID covers patient-organ pair)
        ...


class MELD(Policy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self._setup()

    def _select(self, organ: str) -> int:
        if len(self.waitlist) == 0:
            print(self)
            print("_select", self.waitlist)
            raise Warning("empty waitlist")
        else:
            X = max(self.waitlist, key=attrgetter("time_to_live"))
            self.remove([X])
            return X, organ

    def select(self, organs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.waitlist) == 0:
            print("select", self.waitlist)

            recipients, organs = np.array([]), np.array([])

        elif organs.shape[0] == 0:
            print("organs 0")
            recipients, organs = np.array([]), np.array([])

        elif organs.shape[0] > self.waitlist.shape[0]:
            organs = organs[: len(self.waitlist), :]
            recipients, organs = self.select(organs=organs)

        else:  # organs <= waitlist
            l = np.array(  # noqa E741
                [self._select(organ) for organ in organs], dtype=object
            )
            z = zip(*l)  # noqa E741
            l = list(z)  # noqa E741
            recipients = np.array(l[0])
            organs = np.array(l[1]).reshape(len(recipients), -1)

        return recipients, organs

    def add(self, patients: np.ndarray) -> None:
        if len(patients) == 0:
            return

        MELD_score = self._meld(patients)
        X = [
            Patient(id=p.id, covariates=p.covariates, time_to_live=MELD_score[i])
            for i, p in enumerate(patients)
        ]

        self.waitlist = np.append(self.waitlist, X)

    def _setup(self) -> None:
        MELD_score = self._meld(self.waitlist)

        self.waitlist = np.array(
            [
                Patient(covariates=p.covariates, time_to_live=MELD_score[i], id=p.id)
                for i, p in enumerate(self.waitlist)
            ]
        )

    def _meld(self, patients: np.ndarray) -> np.ndarray:
        ps = pd.DataFrame(columns=self.dm.x_cols, data=[p.covariates for p in patients])

        # DEFINITION OF (standard) MELD: https://en.wikipedia.org/wiki/Model_for_End-Stage_Liver_Disease#Determination
        MELD_score = (
            3.79 * ps.SERUM_BILIRUBIN
            + 11.2 * ps.INR
            + 9.57 * ps.SERUM_CREATININE
            + 6.43
        )

        return MELD_score.to_numpy()

    def score(self, patients: np.ndarray) -> np.ndarray:
        return self._meld(patients)

    def remove(self, x: np.ndarray) -> None:
        for patient in x:
            self.waitlist = np.delete(self.waitlist, np.where(self.waitlist == patient))


class MELD_na(MELD):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

    def _meld(self, patients: np.ndarray) -> np.ndarray:
        # We can simply inherit from MELD as the only part
        # that changes is they way we compute a MELD score
        # by adding the last term in MELD_score.

        ps = pd.DataFrame(columns=self.dm.x_cols, data=[p.covariates for p in patients])
        # ps.loc[:, self.dm.real_cols] = self.dm.scaler.inverse_transform(
        #     ps[self.dm.real_cols]
        # )

        # MELD-na: MELD + 1.59*(135-SODIUM(mmol/l)) (https://github.com/kartoun/meld-plus/raw/master/MELD_Plus_Calculator.xlsx)
        MELD_score = super()._meld(patients) + 1.59 * (135 - ps.SERUM_SODIUM)
        return MELD_score.to_numpy()


# Naive FIFO policy
class FIFO(Policy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

    def remove(self, x: np.ndarray) -> None:
        for patient in x:
            self.waitlist = np.delete(
                self.waitlist, np.where(self.waitlist == patient)[0]
            )

    def add(self, x: np.ndarray) -> None:
        self.waitlist = np.append(self.waitlist, x)

    def select(self, organs: np.ndarray) -> np.ndarray:
        patients = self.waitlist[: len(organs)]
        self.remove(patients)

        return patients, organs.reshape(len(patients), -1)


class MaxPolicy(Policy):
    def __init__(
        self,
        name: str,  # policy name, reported in wandb
        initial_waitlist: np.ndarray,  # waitlist upon starting the simulation, [int]
        dm: OrganDataModule,  # datamodule containing all information of the transplant system
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)

        self._setup()

    def _setup(self) -> None:
        self.x_cols = self.dm.x_cols
        waitlist_patients = self.test.loc[self.waitlist, self.x_cols].copy().to_numpy()

        self.waitlist = np.array(
            [
                Patient(id=self.waitlist[i], covariates=waitlist_patients[i])
                for i in range(len(self.waitlist))
            ]
        )
        self.waitlist = np.unique(self.waitlist)

    def get_xs(self, organs: np.ndarray) -> np.ndarray:
        if len(organs) == 0 or len(self.waitlist) == 0:
            return np.array([])

        if len(organs) > len(self.waitlist):
            for i in range(len(self.waitlist)):
                patient_ids = [
                    self._get_x(organs[i]) for i in range(len(self.waitlist))
                ]
                return patient_ids

        patient_ids = [self._get_x(organ) for organ in organs]

        return patient_ids

    def _get_x(self, organ: int) -> int:
        patient_covariates = np.array([p.covariates for p in self.waitlist])
        organ_covariates = self.test.loc[organ, self.dm.o_cols].to_numpy()

        scores = self._calculate_scores(patient_covariates, [organ_covariates])
        top_index = np.argmax(scores)
        patient_id = self.waitlist[top_index].id
        self.remove_x([patient_id])

        return patient_id

    def add_x(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return

        patient_covariates = self.test.loc[x, self.x_cols].copy().to_numpy()
        patients = [
            Patient(id=x[i], covariates=patient_covariates[i]) for i in range(len(x))
        ]
        self.waitlist = np.append(self.waitlist, patients)
        self.waitlist = np.unique(self.waitlist)

    def remove_x(self, x: np.ndarray) -> None:
        for patient in x:
            self.waitlist = np.array([p for p in self.waitlist if p.id not in x])

    @abstractclassmethod
    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        # this method should return, for each patient in
        # x_covariates, the score of that patient associated
        # with o_covariates. Note that o_covariates is just
        # one organ. This allows to remove the selected patient
        # from the waitlist.
        ...


class OrganITE(MaxPolicy):
    def __init__(
        self,
        name: str,
        initial_waitlist: np.ndarray,
        dm: OrganDataModule,
        inference_ITE: Inference,
        inference_VAE: Inference,
        a: float = 1.0,
        b: float = 1.0,
        data: str = "test",
    ) -> None:
        super().__init__(name, initial_waitlist, dm, data)
        self.inference_ITE = inference_ITE
        self.inference_VAE = inference_VAE

        self.a = a
        self.b = b

    def _setup(self) -> None:
        super()._setup()

        # self.k_means = self.inference_ITE.model.cluster                                 # LOAD CLUSTERS FROM inference_ITE

    def _calculate_scores(
        self, x_covariates: np.ndarray, o_covariates: np.ndarray
    ) -> np.ndarray:
        scores = [
            self._calculate_score(
                np.array([patient]), np.array(o_covariates, dtype=float)
            )
            for patient in x_covariates
        ]

        return scores

    def _calculate_score(self, patient: np.ndarray, organ: np.ndarray) -> np.ndarray:
        ITE = self.inference_ITE(patient, organ)

        ITE *= self._get_lambda(patient, organ)

        return ITE

    def _get_optimal_organ(self, patient: np.ndarray) -> np.ndarray:
        sample_organs = self.dm._train_processed.sample(n=512)[
            self.dm.o_cols
        ].to_numpy()
        repeated_patients = np.repeat(patient, 512, axis=0)
        ITEs = self.inference_ITE(repeated_patients, sample_organs)
        optimal_organ_ix = np.argmax(ITEs)
        optimal_organ = sample_organs[optimal_organ_ix]

        return optimal_organ.reshape(1, -1)

    def _get_lambda(self, patient: np.ndarray, organ: np.ndarray) -> np.ndarray:
        optimal_organ = self._get_optimal_organ(patient)
        propensity = self._get_propensities([optimal_organ])
        distance = self._get_distances(optimal_organ, organ)

        lam = ((propensity + 0.000001) ** (-self.a)) * (
            distance + 0.000001 ** (-self.b)
        )
        return lam

    def _get_distances(self, organ_A: np.ndarray, organ_B: np.ndarray) -> np.ndarray:
        distance = scipy.spatial.distance.euclidean(organ_A, organ_B)
        return distance

    def _get_ITE(self, organ: np.ndarray) -> np.ndarray:
        patients = np.array([p.covariates for p in self.waitlist])
        organs = np.repeat(organ, len(patients), axis=0)
        null_organs = np.zeros(organs.shape)

        Y_1 = self.inference_ITE(patients, organs)
        Y_0 = self.inference_ITE(patients, null_organs)

        return (Y_1 - Y_0).numpy()

    def _get_propensities(
        self,
        o_covariates: np.ndarray,
    ) -> np.ndarray:
        return self.inference_VAE(torch.Tensor(o_covariates).double())

    def _get_patients(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        return self._get_instances(x, self.dm.x_cols, data_class=Patient, train=train)

    def _get_organs(
        self, o: np.ndarray, organ: np.ndarray, train: bool = False
    ) -> np.ndarray:
        return self._get_instances(o, self.dm.o_cols, data_class=organ, train=train)
        # return self._get_instances(o, self.dm.o_cols, data_class=Organ (undefined), train=train)

    def _get_instances(
        self,
        l: np.ndarray,
        cols: np.ndarray,
        data_class: Any,
        train: bool = False,
    ) -> np.ndarray:
        data = self.test
        if train:
            data = self.dm._train_processed
        covariates = data.loc[data.index.isin(l), cols].copy()
        types = np.array(
            [
                data_class(id=l[i], covariates=covariates.iloc[i].to_numpy())
                for i in range(len(l))
            ]
        )

        return types
