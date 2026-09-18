"""Value objects do gerador de TLEs sintéticos.

Este módulo é deliberadamente livre de dependências: só stdlib. Nada de
numpy, sgp4, passpredict ou orbit_predictor. Se um import dessas
bibliotecas aparecer aqui, alguma responsabilidade vazou pro lugar errado.

Convenção de unidades: o núcleo trabalha em radianos e SI. Graus só
existem nas fronteiras -- na entrada (GroundPoint, que é o que o usuário
digita) e na saída (formatter, que escreve as colunas do TLE).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace

__all__ = [
    "GroundPoint",
    "OrbitalElements",
    "TleIdentity",
    "PassSolution",
]


@dataclass(frozen=True, slots=True)
class GroundPoint:
    """Ponto fixo na superfície da Terra, em coordenadas geodésicas."""

    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError(f"lat_deg fora de [-90, 90]: {self.lat_deg}")
        if not -360.0 <= self.lon_deg <= 360.0:
            raise ValueError(f"lon_deg fora de [-360, 360]: {self.lon_deg}")

    @property
    def lat_rad(self) -> float:
        return math.radians(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        return math.radians(self.lon_deg)


@dataclass(frozen=True, slots=True)
class OrbitalElements:
    """Elementos keplerianos médios, em radianos e SI.

    Não guarda o semi-eixo maior de propósito: `a` e `mean_motion` são
    redundantes, e manter os dois abre espaço pra ficarem inconsistentes.
    A conversão entre eles precisa de mu, que é física -- mora em
    geometry.py, não aqui.
    """

    inc_rad: float
    raan_rad: float
    ecc: float
    argp_rad: float
    mean_anomaly_rad: float
    mean_motion_rad_s: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.inc_rad <= math.pi:
            raise ValueError(f"inc_rad fora de [0, pi]: {self.inc_rad}")
        if not 0.0 <= self.ecc < 1.0:
            raise ValueError(f"ecc fora de [0, 1): {self.ecc}")
        if self.mean_motion_rad_s <= 0.0:
            raise ValueError(
                f"mean_motion_rad_s precisa ser positivo: {self.mean_motion_rad_s}"
            )

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion_rad_s

    @property
    def mean_motion_rev_per_day(self) -> float:
        """Mean motion no formato que a coluna 53-63 do TLE espera."""
        return 86400.0 / self.period_s

    def with_mean_motion(self, mean_motion_rad_s: float) -> OrbitalElements:
        """Cópia com outro mean motion -- usado pela calibração de Kozai."""
        return replace(self, mean_motion_rad_s=mean_motion_rad_s)


@dataclass(frozen=True, slots=True)
class TleIdentity:
    """Campos do TLE que são burocracia de formato, não física."""

    satnum: int = 99999
    classification: str = "U"
    intl_designator: str = "99999A"
    element_set: int = 999
    rev_number: int = 1
    name: str = "FAKE-SAT"

    def __post_init__(self) -> None:
        if not 1 <= self.satnum <= 99999:
            raise ValueError(f"satnum precisa caber em 5 dígitos: {self.satnum}")
        if self.classification not in ("U", "C", "S"):
            raise ValueError(f"classification inválida: {self.classification!r}")
        if len(self.intl_designator) > 8:
            raise ValueError(
                f"intl_designator excede 8 colunas: {self.intl_designator!r}"
            )
        if not 0 <= self.element_set <= 9999:
            raise ValueError(
                f"element_set precisa caber em 4 dígitos: {self.element_set}"
            )
        if not 0 <= self.rev_number <= 99999:
            raise ValueError(
                f"rev_number precisa caber em 5 dígitos: {self.rev_number}"
            )


@dataclass(frozen=True, slots=True)
class PassSolution:
    """Resultado do solver: a órbita, mais quando ela chega no alvo."""

    elements: OrbitalElements
    epoch: dt.datetime
    time_of_flight_s: float
    iterations: int

    def __post_init__(self) -> None:
        if self.epoch.tzinfo is None or self.epoch.utcoffset() is None:
            raise ValueError(
                "epoch precisa ser timezone-aware; datetime ingênuo leva a GMST errado"
            )
        if self.time_of_flight_s < 0.0:
            raise ValueError(
                f"time_of_flight_s não pode ser negativo: {self.time_of_flight_s}"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations precisa ser >= 1: {self.iterations}")

    @property
    def arrival(self) -> dt.datetime:
        """Instante em que o satélite passa sobre o ponto alvo."""
        return self.epoch + dt.timedelta(seconds=self.time_of_flight_s)

    @property
    def revolutions(self) -> float:
        """Quantas voltas o satélite dá entre a época e a chegada."""
        return self.time_of_flight_s / self.elements.period_s
