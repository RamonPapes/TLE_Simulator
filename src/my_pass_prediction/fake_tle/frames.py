"""Implementações concretas de `SiderealTime` e `EarthModel`.

Único módulo do pacote que conhece o orbit_predictor. Trocar de
biblioteca de tempo sideral mexe aqui e em lugar nenhum.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from orbit_predictor.utils import gstime_from_datetime

from .models import GroundPoint

__all__ = ["Iau82SiderealTime", "SphericalEarth", "Wgs84Earth"]

_TWO_PI = 2.0 * math.pi

# WGS84
_A_EARTH_KM = 6378.137
_FLATTENING = 1.0 / 298.257_223_563
_ECC_SQ = _FLATTENING * (2.0 - _FLATTENING)


class Iau82SiderealTime:
    """GMST pelo modelo IAU-82, via orbit_predictor."""

    def gmst_rad(self, when: dt.datetime) -> float:
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("`when` precisa ser timezone-aware")
        # gstime_from_datetime lê os campos de parede e IGNORA o tzinfo:
        # sem este astimezone, um datetime em outro fuso dá GMST errado
        # sem qualquer aviso.
        return gstime_from_datetime(when.astimezone(dt.timezone.utc)) % _TWO_PI


class SphericalEarth:
    """Terra esférica: rápido e suficiente pra estudo de distância.

    Ignora o achatamento, então a latitude geocêntrica difere da
    geodésica em até 0.19 graus (~21 km no solo). Use `Wgs84Earth` se
    isso importar.
    """

    def to_eci_unit(self, point: GroundPoint, gmst_rad: float) -> np.ndarray:
        right_ascension = gmst_rad + point.lon_rad
        lat = point.lat_rad
        return np.array(
            [
                math.cos(lat) * math.cos(right_ascension),
                math.cos(lat) * math.sin(right_ascension),
                math.sin(lat),
            ]
        )


class Wgs84Earth:
    """Elipsoide WGS84: a latitude de entrada é tratada como geodésica."""

    def to_eci_unit(self, point: GroundPoint, gmst_rad: float) -> np.ndarray:
        lat, alt_km = point.lat_rad, point.alt_m / 1000.0
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        # Raio de curvatura no primeiro vertical.
        prime_vertical = _A_EARTH_KM / math.sqrt(1.0 - _ECC_SQ * sin_lat**2)
        r_xy = (prime_vertical + alt_km) * cos_lat
        r_z = (prime_vertical * (1.0 - _ECC_SQ) + alt_km) * sin_lat

        right_ascension = gmst_rad + point.lon_rad
        ecef = np.array(
            [r_xy * math.cos(right_ascension), r_xy * math.sin(right_ascension), r_z]
        )
        return ecef / float(np.linalg.norm(ecef))
