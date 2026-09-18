"""Fixtures compartilhadas.

A mais valiosa é `frozen_sidereal`: com a Terra parada, o resultado da
geometria é calculável à mão e os testes viram asserções exatas em vez de
comparações com tolerância arbitrária.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pytest

from my_pass_prediction.fake_tle import (
    GroundPoint,
    Iau82SiderealTime,
    OverheadPassSolver,
    SphericalEarth,
    Wgs84Earth,
)

DATA_DIR = pathlib.Path(__file__).parent / "data"


class FrozenSiderealTime:
    """Stub de `SiderealTime`: congela a rotação da Terra."""

    def __init__(self, gmst_rad: float = 0.0) -> None:
        self._gmst_rad = gmst_rad

    def gmst_rad(self, when: dt.datetime) -> float:
        return self._gmst_rad


@pytest.fixture
def frozen_sidereal() -> FrozenSiderealTime:
    return FrozenSiderealTime()


@pytest.fixture
def real_sidereal() -> Iau82SiderealTime:
    return Iau82SiderealTime()


@pytest.fixture
def spherical_earth() -> SphericalEarth:
    return SphericalEarth()


@pytest.fixture
def wgs84_earth() -> Wgs84Earth:
    return Wgs84Earth()


@pytest.fixture
def epoch() -> dt.datetime:
    return dt.datetime(2026, 9, 18, 12, 0, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def frozen_solver(frozen_sidereal, spherical_earth) -> OverheadPassSolver:
    """Solver determinístico: Terra parada e esférica."""
    return OverheadPassSolver(frozen_sidereal, spherical_earth)


@pytest.fixture
def real_solver(real_sidereal, wgs84_earth) -> OverheadPassSolver:
    return OverheadPassSolver(real_sidereal, wgs84_earth)


@pytest.fixture
def iss_lines() -> tuple[str, str]:
    """TLE real da ISS, congelado em arquivo -- nenhum teste vai à rede."""
    line1, line2 = (DATA_DIR / "iss_reference.tle").read_text().strip().splitlines()
    return line1, line2


@pytest.fixture
def salvador() -> GroundPoint:
    return GroundPoint(-12.9777, -38.5016)


@pytest.fixture
def sao_paulo() -> GroundPoint:
    return GroundPoint(-23.5505, -46.6333)
