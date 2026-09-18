"""Geometria orbital pura: vetores, planos e ângulos.

Funções sem estado, sem tempo e sem I/O -- é o módulo mais fácil de testar
do pacote. Também é o único lugar onde mu e o raio da Terra aparecem.

Tudo em radianos e km.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MU_EARTH_KM3_S2",
    "R_EARTH_KM",
    "DegenerateGeometryError",
    "mean_motion_from_altitude",
    "semi_major_axis_from_mean_motion",
    "normalize",
    "plane_from_directions",
    "inclination_raan",
    "ascending_node_vector",
    "project_onto_plane",
    "argument_of_latitude",
    "rotate_about_axis",
    "central_angle_for_elevation",
]

# WGS84. O valor exato importa pouco: a calibração de Kozai em
# generator.py corrige a diferença residual contra o próprio SGP4.
MU_EARTH_KM3_S2 = 398_600.4418
R_EARTH_KM = 6378.137

_TOL = 1e-9
_TWO_PI = 2.0 * math.pi


class DegenerateGeometryError(ValueError):
    """Os dois pontos não definem um plano orbital único."""


def mean_motion_from_altitude(altitude_km: float) -> float:
    """Mean motion kepleriano de uma órbita circular, em rad/s."""
    if altitude_km <= -R_EARTH_KM:
        raise ValueError(f"altitude_km inviável: {altitude_km}")
    a = R_EARTH_KM + altitude_km
    return math.sqrt(MU_EARTH_KM3_S2 / a**3)


def semi_major_axis_from_mean_motion(mean_motion_rad_s: float) -> float:
    """Semi-eixo maior em km. Inverso de `mean_motion_from_altitude`."""
    if mean_motion_rad_s <= 0.0:
        raise ValueError(f"mean_motion_rad_s precisa ser positivo: {mean_motion_rad_s}")
    return (MU_EARTH_KM3_S2 / mean_motion_rad_s**2) ** (1.0 / 3.0)


def normalize(vec) -> np.ndarray:
    """Vetor unitário. Levanta se o vetor for (quase) nulo."""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < _TOL:
        raise DegenerateGeometryError("vetor nulo não tem direção definida")
    return arr / norm


def plane_from_directions(u_a, u_b, retrograde: bool = False) -> np.ndarray:
    """Momento angular unitário do plano que contém as duas direções.

    Duas direções e o centro da Terra definem um plano; `h_hat` é a normal
    dele. O sinal escolhe o sentido de percurso: com `retrograde=False` o
    satélite vai de A para B pelo arco curto.
    """
    cross = np.cross(np.asarray(u_a, float), np.asarray(u_b, float))
    if float(np.linalg.norm(cross)) < _TOL:
        raise DegenerateGeometryError(
            "pontos coincidentes ou antípodas: infinitos planos passam pelos dois"
        )
    h_hat = cross / float(np.linalg.norm(cross))
    return -h_hat if retrograde else h_hat


def inclination_raan(h_hat) -> tuple[float, float]:
    """Inclinação e RAAN, em radianos, a partir de `h_hat`.

    Vem de h_hat = [sin i sin RAAN, -sin i cos RAAN, cos i].
    """
    h = np.asarray(h_hat, float)
    inc = math.acos(min(1.0, max(-1.0, float(h[2]))))
    if abs(h[0]) < _TOL and abs(h[1]) < _TOL:
        # Órbita equatorial: o nó ascendente é indefinido. Convenção: 0.
        return inc, 0.0
    return inc, math.atan2(float(h[0]), -float(h[1])) % _TWO_PI


def ascending_node_vector(h_hat) -> np.ndarray:
    """Direção unitária do nó ascendente (z_hat x h_hat, normalizado)."""
    h = np.asarray(h_hat, float)
    node = np.array([-h[1], h[0], 0.0])
    norm = float(np.linalg.norm(node))
    if norm < _TOL:
        # Equatorial: casa com a convenção RAAN = 0 de `inclination_raan`.
        return np.array([1.0, 0.0, 0.0])
    return node / norm


def project_onto_plane(vec, h_hat) -> np.ndarray:
    """Projeção unitária de `vec` no plano orbital normal a `h_hat`.

    Necessária quando o alvo não está exatamente sob a ground track (ver
    `MaxElevationTarget`): o que interessa é o instante de máxima
    aproximação, que corresponde à projeção.
    """
    v = np.asarray(vec, float)
    h = np.asarray(h_hat, float)
    return normalize(v - h * float(np.dot(h, v)))


def argument_of_latitude(h_hat, r_hat) -> float:
    """Ângulo do nó ascendente até `r_hat`, no sentido do movimento.

    Com e = 0 e argp = 0 esse ângulo *é* a anomalia média.
    """
    node = ascending_node_vector(h_hat)
    quarter_turn = np.cross(np.asarray(h_hat, float), node)
    r = np.asarray(r_hat, float)
    return math.atan2(float(np.dot(quarter_turn, r)), float(np.dot(node, r))) % _TWO_PI


def rotate_about_axis(vec, axis_hat, angle_rad: float) -> np.ndarray:
    """Rotação de Rodrigues de `vec` em torno de `axis_hat`."""
    v = np.asarray(vec, float)
    k = np.asarray(axis_hat, float)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return v * cos_a + np.cross(k, v) * sin_a + k * float(np.dot(k, v)) * (1.0 - cos_a)


def central_angle_for_elevation(elevation_deg: float, altitude_km: float) -> float:
    """Ângulo central entre o ponto no solo e o subsatélite, em radianos.

    Da lei dos senos no triângulo centro-estação-satélite:
        cos(E + gamma) = (R / r) * cos(E)
    Em 500 km: E = 90 graus -> gamma = 0; E = 0 -> gamma = 22 graus.
    """
    if not 0.0 <= elevation_deg <= 90.0:
        raise ValueError(f"elevation_deg fora de [0, 90]: {elevation_deg}")
    elev = math.radians(elevation_deg)
    r = R_EARTH_KM + altitude_km
    return math.acos(min(1.0, max(-1.0, (R_EARTH_KM / r) * math.cos(elev)))) - elev
