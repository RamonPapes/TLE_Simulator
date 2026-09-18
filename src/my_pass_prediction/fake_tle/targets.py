"""Estratégias de escolha do plano orbital -- o ponto de extensão (OCP).

Um requisito novo do tipo "quero que passe a 40 graus de elevação sobre
B" vira uma classe nova aqui; o solver não muda.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .geometry import (
    DegenerateGeometryError,
    central_angle_for_elevation,
    plane_from_directions,
    rotate_about_axis,
)

__all__ = ["PlaneTarget", "OverheadTarget", "MaxElevationTarget"]


@runtime_checkable
class PlaneTarget(Protocol):
    """Escolhe o plano orbital (via `h_hat`) que satisfaz o alvo."""

    def constrain(self, u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class OverheadTarget:
    """Passagem pelo zênite: o plano contém os dois pontos."""

    retrograde: bool = False

    def constrain(self, u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
        return plane_from_directions(u_a, u_b, self.retrograde)


@dataclass(frozen=True, slots=True)
class MaxElevationTarget:
    """Passagem com elevação máxima dada sobre o alvo.

    Parte do plano do zênite e o rotaciona em torno de `u_a`. Essa
    rotação preserva `dot(h_hat, u_a) = 0`, ou seja: a origem continua
    exatamente sob a ground track e só o alvo fica deslocado.

    `offset_sign` escolhe de que lado da ground track o alvo cai.
    """

    elevation_deg: float
    altitude_km: float = 500.0
    retrograde: bool = False
    offset_sign: int = 1

    def __post_init__(self) -> None:
        if self.offset_sign not in (-1, 1):
            raise ValueError(f"offset_sign precisa ser -1 ou 1: {self.offset_sign}")

    def constrain(self, u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
        h_overhead = plane_from_directions(u_a, u_b, self.retrograde)
        gamma = central_angle_for_elevation(self.elevation_deg, self.altitude_km)
        if gamma < 1e-12:
            return h_overhead

        # Rodrigues em torno de u_a, com h ortogonal a u_a, reduz a
        #     dot(R(alpha) h, u_b) = sin(alpha) * dot(u_a x h, u_b)
        # e esse produto escalar vale -sen(separação angular A-B).
        swing = np.cross(u_a, h_overhead)
        denominator = float(np.dot(swing, u_b))
        sin_alpha = self.offset_sign * math.sin(gamma) / denominator
        if abs(sin_alpha) > 1.0:
            raise DegenerateGeometryError(
                f"elevação de {self.elevation_deg} graus exige um desvio maior "
                "que a separação angular entre os dois pontos"
            )
        return rotate_about_axis(h_overhead, u_a, math.asin(sin_alpha))
