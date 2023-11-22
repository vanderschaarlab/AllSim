from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from allsim.infer import System
from allsim.outcome.counterfactual_inference import Inference
from allsim.policies import Policy
from allsim.policies.base import Patient, Stats

# SIMULATION OVERVIEW:
#   1. -> setup waitlist of patients
#   2. -> setup available organs (amount
#       will be fraction of patients)
#   3. -> shuffle patients df and organs df
#   4. -> iterate over patients:
#       -> sample patient(s)
#       -> sample organ(s)
#       -> remove dead patients from waitlist (also in policy)
#       -> update statistics


# SIM:
#   Sim.fit(
#       patients[X1, ..., Xd, idX, t],
#       resources[R1, ..., Re, t, idR],
#       matches[idX, idR, Y]) -> fit densities, arrival_rates, outcome_model

#   Sim.run(time_units; optional)


class Sim:
    def __init__(
        self,
        resource_system: System,
        patient_system: System,
        inference: Inference,
    ) -> None:
        self.resource_system = resource_system
        self.patient_system = patient_system
        self.inference = inference

        self._internal_waitlist = np.array([])
        self.log_df = pd.DataFrame(
            columns=[
                *resource_system.density.columns,
                *patient_system.density.columns,
                "Y",
            ]
        )

        self._setup()

    def _setup(self, initial_waitlist_size: int = 100):
        # FILL WAITLIST WITH initial_waitlist_size PATIENTS
        patients = self.patient_system.density.sample(
            n=initial_waitlist_size
        ).to_numpy()

        wl = np.array(
            [
                Patient(
                    covariates=p,
                    id=-i,
                    time_to_live=self.inference(x=p.reshape(1, -1), r=None).item(),
                )
                for i, p in enumerate(patients)
            ]
        )

        self._internal_waitlist = np.append(self._internal_waitlist, wl, axis=0)

    def simulate(self, policy: Policy, T: int = 365) -> Tuple[Stats, pd.DataFrame]:
        for t in tqdm(range(1, T)):
            self.iterate(policy=policy, t=t)

        return self.log_df

    def iterate(self, policy: Policy, t: int):
        data = np.empty(
            (
                0,
                len(self.patient_system.density.columns)
                + len(self.resource_system.density.columns)
                + 2,
            )  # + ttl + t
        )

        dead_patients = self._remove_dead_patients(policy)
        if len(dead_patients):
            dead_patients = np.array([p.covariates for p in dead_patients])
            dead_patients = dead_patients.reshape(len(dead_patients), -1)
            time = np.repeat(t, len(dead_patients)).reshape(-1, 1)
            ttl = self.inference(x=dead_patients, r=None).reshape(-1, 1)
            dead_patients = np.append(
                dead_patients,
                np.full(
                    (len(dead_patients), len(self.resource_system.density.columns)),
                    np.nan,
                ),
                axis=1,
            )
            dead_patients = np.append(dead_patients, time, axis=1)
            dead_patients = np.append(dead_patients, ttl, axis=1)
        else:
            dead_patients = np.empty(data.shape)

        patients = self._sample_patients(t=t)
        policy.add(patients)

        resources = self._sample_resources(t=t)

        if len(resources):
            recipients, resources = policy.select(resources.to_numpy())
            recipient_covariates = np.array([p.covariates for p in recipients])

            ttl = self.inference(x=recipient_covariates, r=resources).flatten()

            recipient_data = np.append(recipient_covariates, resources, axis=1)
            recipient_data = np.append(
                recipient_data, np.repeat(t, len(recipient_data)).reshape(-1, 1), axis=1
            )
            recipient_data = np.append(recipient_data, ttl.reshape(-1, 1), axis=1)

            self._remove_patients(recipients)
        else:
            recipient_data = np.empty(data.shape)

        data = np.append(data, dead_patients, axis=0)
        data = np.append(data, recipient_data, axis=0)

        log = pd.DataFrame(
            columns=[
                *self.patient_system.density.columns,
                *self.resource_system.density.columns,
                "t",
                "ttl",
            ],
            data=data,
        )

        self.log_df = self.log_df.append(log)

        self._age_patients(days=1)

    def _remove_patients(self, patients: list) -> None:
        self._internal_waitlist = np.delete(
            self._internal_waitlist,
            np.intersect1d(
                np.array([p.id for p in self._internal_waitlist]),
                [p.id for p in patients],
                return_indices=True,
            )[1],
        )

    def _remove_dead_patients(self, policy: Policy) -> list:
        dead_patients_indices = np.where(
            np.array([p.time_to_live for p in self._internal_waitlist]) <= 0
        )[
            0
        ]  # selects patient IDs when Sim_Patient.ttl <= 0

        dead_patients = self._internal_waitlist[dead_patients_indices]
        self._remove_patients(dead_patients)  # remove patients from self.waitlist
        policy.remove(dead_patients)
        return dead_patients

    def _age_patients(self, days: int = 1) -> None:
        self._internal_waitlist = np.array(
            [p.age(days) for p in self._internal_waitlist]
        )

    def _sample_patients(self, t: int) -> np.ndarray:
        patients = self.patient_system(t)
        if len(patients) == 0:
            return np.empty((0, len(self.patient_system.density.columns)))

        patients = patients.to_numpy().reshape(len(patients), -1)
        ttl = self.inference(x=patients, r=None).flatten()

        patients = [
            Patient(covariates=patient, time_to_live=ttl[i], id=i * t)
            for i, patient in enumerate(patients)
        ]

        self._internal_waitlist = np.append(self._internal_waitlist, patients)

        return patients

    def _sample_resources(self, t: int) -> np.ndarray:
        return self.resource_system(t)
