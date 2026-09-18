"""Composition root: junta solver, calibração e formatter num TLE.

Único módulo que importa passpredict.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from orbit_predictor.coordinate_systems import ecef_to_llh
from passpredict import TLE, SGP4Propagator

from .compat import PatchedTLE

from .formatter import format_tle
from .geometry import (
    R_EARTH_KM,
    mean_motion_from_altitude,
    semi_major_axis_from_mean_motion,
)
from .models import GroundPoint, OrbitalElements, PassSolution, TleIdentity
from .protocols import AltitudeProbe
from .solver import ConvergenceError, OverheadPassSolver
from .targets import PlaneTarget

__all__ = ["Sgp4AltitudeProbe", "KozaiCalibrator", "FakeTleGenerator", "subpoint"]


def subpoint(
    satellite: SGP4Propagator, when: dt.datetime
) -> tuple[float, float, float]:
    """Ponto subsatélite geodésico: (lat_deg, lon_deg, alt_km).

    `SGP4Propagator.get_llh` levanta NotImplementedError nesta versão do
    passpredict, então o caminho é ECEF -> geodésica na mão.
    """
    lat, lon, alt = ecef_to_llh(satellite.get_only_position(when))
    return float(lat), float(lon), float(alt)


class Sgp4AltitudeProbe:
    """Mede, propagando com SGP4, a altitude que os elementos produzem."""

    def __init__(self, identity: TleIdentity | None = None, samples: int = 32) -> None:
        if samples < 1:
            raise ValueError(f"samples precisa ser >= 1: {samples}")
        self._identity = identity or TleIdentity()
        self._samples = samples

    def mean_altitude_km(
        self, elements: OrbitalElements, epoch: dt.datetime
    ) -> float:
        """Altitude geocêntrica média (|r| - R_EARTH_KM) sobre uma revolução.

        Geocêntrica, e não geodésica, de propósito: a altitude sobre o
        elipsoide depende da latitude, então numa órbita inclinada ela
        misturaria o achatamento da Terra com a correção de Kozai, que é
        o que se quer isolar aqui.
        """
        line1, line2 = format_tle(elements, epoch, self._identity)
        satellite = SGP4Propagator.from_tle(
            PatchedTLE(
                self._identity.satnum, (line1, line2), name=self._identity.name
            )
        )
        step = elements.period_s / self._samples
        radii = [
            float(
                np.linalg.norm(
                    satellite.get_only_position(
                        epoch + dt.timedelta(seconds=i * step)
                    )
                )
            )
            for i in range(self._samples)
        ]
        return float(np.mean(radii)) - R_EARTH_KM


class KozaiCalibrator:
    """Corrige o mean motion pra bater a altitude alvo no SGP4.

    O mean motion do TLE é o de Kozai, não o kepleriano: o SGP4 desfaz a
    transformação internamente (ver `orbit_predictor.utils.unkozai`), e o
    J2 desloca a altitude propagada em alguns km. Em vez de inverter a
    relação analiticamente, mede e ajusta.
    """

    def __init__(
        self,
        probe: AltitudeProbe,
        tol_km: float = 0.1,
        max_iter: int = 10,
    ) -> None:
        self._probe = probe
        self._tol_km = tol_km
        self._max_iter = max_iter

    def calibrate(
        self,
        elements: OrbitalElements,
        epoch: dt.datetime,
        target_altitude_km: float,
    ) -> OrbitalElements:
        current = elements
        for _ in range(self._max_iter):
            error_km = target_altitude_km - self._probe.mean_altitude_km(current, epoch)
            if abs(error_km) < self._tol_km:
                return current
            semi_major_axis = semi_major_axis_from_mean_motion(
                current.mean_motion_rad_s
            )
            current = current.with_mean_motion(
                mean_motion_from_altitude(semi_major_axis + error_km - R_EARTH_KM)
            )
        raise ConvergenceError(
            f"calibração de Kozai não convergiu em {self._max_iter} iterações "
            f"(erro residual {error_km:.3f} km)"
        )


class FakeTleGenerator:
    """Orquestra solver -> calibração -> formatter."""

    def __init__(
        self,
        solver: OverheadPassSolver,
        identity: TleIdentity | None = None,
        calibrator: KozaiCalibrator | None = None,
    ) -> None:
        self._solver = solver
        self._identity = identity or TleIdentity()
        self._calibrator = calibrator

    def generate(
        self,
        origin: GroundPoint,
        target: GroundPoint,
        epoch: dt.datetime,
        altitude_km: float = 500.0,
        plane_target: PlaneTarget | None = None,
        revolutions: int = 0,
    ) -> tuple[TLE, PassSolution]:
        solution = self._solver.solve(
            origin,
            target,
            epoch,
            altitude_km=altitude_km,
            plane_target=plane_target,
            revolutions=revolutions,
        )

        if self._calibrator is not None:
            calibrated = self._calibrator.calibrate(
                solution.elements, epoch, altitude_km
            )
            # Re-resolver: o n corrigido muda o período e portanto o
            # tempo de voo. Sem isso a chegada erra alguns segundos.
            solution = self._solver.solve(
                origin,
                target,
                epoch,
                altitude_km=altitude_km,
                plane_target=plane_target,
                revolutions=revolutions,
                mean_motion_rad_s=calibrated.mean_motion_rad_s,
            )

        line1, line2 = format_tle(solution.elements, epoch, self._identity)
        tle = PatchedTLE(
            self._identity.satnum, (line1, line2), name=self._identity.name
        )
        return tle, solution
