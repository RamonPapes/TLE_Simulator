"""Gerador de TLEs sintéticos para simulação de passagens.

Fixa um ponto de origem (sob o satélite na época) e um ponto alvo por
onde ele precisa passar, numa órbita circular LEO.

    >>> import datetime as dt
    >>> from my_pass_prediction.fake_tle import GroundPoint, generate_fake_tle
    >>> salvador = GroundPoint(-12.9777, -38.5016)
    >>> sao_paulo = GroundPoint(-23.5505, -46.6333)
    >>> epoch = dt.datetime(2026, 9, 18, 12, 0, tzinfo=dt.timezone.utc)
    >>> tle, solution = generate_fake_tle(salvador, sao_paulo, epoch)

Para injetar dependências (outro modelo de Terra, outro alvo de plano),
use `FakeTleGenerator` e `OverheadPassSolver` diretamente.
"""

from __future__ import annotations

import datetime as dt

from passpredict import TLE

from .compat import PatchedTLE
from .frames import Iau82SiderealTime, SphericalEarth, Wgs84Earth
from .generator import (
    FakeTleGenerator,
    KozaiCalibrator,
    Sgp4AltitudeProbe,
    subpoint,
)
from .geometry import DegenerateGeometryError, point_at_bearing
from .models import GroundPoint, OrbitalElements, PassSolution, TleIdentity
from .protocols import AltitudeProbe, EarthModel, SiderealTime
from .solver import ConvergenceError, OverheadPassSolver
from .targets import MaxElevationTarget, OverheadTarget, PlaneTarget

__all__ = [
    "generate_fake_tle",
    "GroundPoint",
    "OrbitalElements",
    "PassSolution",
    "TleIdentity",
    "OverheadPassSolver",
    "ConvergenceError",
    "DegenerateGeometryError",
    "FakeTleGenerator",
    "KozaiCalibrator",
    "Sgp4AltitudeProbe",
    "OverheadTarget",
    "MaxElevationTarget",
    "PlaneTarget",
    "Iau82SiderealTime",
    "SphericalEarth",
    "Wgs84Earth",
    "SiderealTime",
    "EarthModel",
    "AltitudeProbe",
    "subpoint",
    "point_at_bearing",
    "PatchedTLE",
]


def generate_fake_tle(
    origin: GroundPoint,
    target: GroundPoint,
    epoch: dt.datetime,
    altitude_km: float = 500.0,
    plane_target: PlaneTarget | None = None,
    revolutions: int = 0,
    identity: TleIdentity | None = None,
    calibrate: bool = True,
) -> tuple[TLE, PassSolution]:
    """Monta as dependências padrão e gera o TLE.

    Padrões: elipsoide WGS84, GMST IAU-82, passagem pelo zênite e
    calibração de Kozai ligada.
    """
    identity = identity or TleIdentity()
    solver = OverheadPassSolver(Iau82SiderealTime(), Wgs84Earth())
    calibrator = KozaiCalibrator(Sgp4AltitudeProbe(identity)) if calibrate else None
    generator = FakeTleGenerator(solver, identity, calibrator)
    return generator.generate(
        origin,
        target,
        epoch,
        altitude_km=altitude_km,
        plane_target=plane_target,
        revolutions=revolutions,
    )
