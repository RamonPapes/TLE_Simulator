"""As costuras de inversão de dependência.

São três, e todas existem por um motivo concreto de teste:

- `SiderealTime`: um stub que devolve sempre 0 congela a rotação da Terra
  e torna a geometria conferível no papel.
- `EarthModel`: permite trocar esfera por WGS84 sem tocar no solver.
- `AltitudeProbe`: permite calibrar Kozai sem importar o SGP4 no teste.

Protocol e não ABC de propósito: as implementações não herdam nada e o
stub de teste é uma classe de três linhas.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import numpy as np

from .models import GroundPoint, OrbitalElements

__all__ = ["SiderealTime", "EarthModel", "AltitudeProbe"]


@runtime_checkable
class SiderealTime(Protocol):
    """Converte um instante no tempo sideral médio de Greenwich."""

    def gmst_rad(self, when: dt.datetime) -> float:
        """GMST em radianos, em [0, 2*pi)."""
        ...


@runtime_checkable
class EarthModel(Protocol):
    """Converte um ponto no solo na direção unitária dele em ECI."""

    def to_eci_unit(self, point: GroundPoint, gmst_rad: float) -> np.ndarray:
        """Vetor unitário (3,) no referencial inercial."""
        ...


@runtime_checkable
class AltitudeProbe(Protocol):
    """Mede a altitude que um conjunto de elementos realmente produz."""

    def mean_altitude_km(
        self, elements: OrbitalElements, epoch: dt.datetime
    ) -> float:
        """Altitude média sobre uma revolução, em km."""
        ...
