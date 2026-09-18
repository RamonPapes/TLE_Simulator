"""O laço de ponto fixo que resolve a geometria da passagem.

O problema: pra saber onde o alvo está em ECI no instante da chegada
preciso do tempo de voo, que depende do arco percorrido, que depende de
onde o alvo está. Circular -- daí a iteração.
"""

from __future__ import annotations

import datetime as dt
import math

from .geometry import (
    argument_of_latitude,
    inclination_raan,
    mean_motion_from_altitude,
    project_onto_plane,
)
from .models import GroundPoint, OrbitalElements, PassSolution
from .protocols import EarthModel, SiderealTime
from .targets import OverheadTarget, PlaneTarget

__all__ = ["ConvergenceError", "OverheadPassSolver", "CIRCULAR_ECC"]

_TWO_PI = 2.0 * math.pi

# Circular de verdade seria e = 0, mas alguns caminhos do SGP4 ficam
# numericamente instáveis com excentricidade exatamente nula. 1e-7 é o
# menor valor representável nas 7 colunas do TLE.
CIRCULAR_ECC = 1e-7


class ConvergenceError(RuntimeError):
    """O laço de ponto fixo não convergiu dentro de `max_iter`."""


class OverheadPassSolver:
    """Resolve os elementos orbitais a partir de dois pontos no solo.

    Única classe com estado do pacote, porque é a única que recebe
    dependências injetadas.
    """

    def __init__(
        self,
        sidereal_time: SiderealTime,
        earth: EarthModel,
        tol_s: float = 0.01,
        max_iter: int = 50,
        relaxation: float = 1.0,
    ) -> None:
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(f"relaxation fora de (0, 1]: {relaxation}")
        self._sidereal_time = sidereal_time
        self._earth = earth
        self._tol_s = tol_s
        self._max_iter = max_iter
        self._relaxation = relaxation

    def solve(
        self,
        origin: GroundPoint,
        target: GroundPoint,
        epoch: dt.datetime,
        altitude_km: float = 500.0,
        plane_target: PlaneTarget | None = None,
        revolutions: int = 0,
        mean_motion_rad_s: float | None = None,
    ) -> PassSolution:
        """Órbita circular que está sobre `origin` na época e passa por `target`.

        `revolutions` seleciona em que volta a chegada acontece: 0 é a
        primeira passagem, 1 uma órbita depois, e assim por diante.

        `mean_motion_rad_s` sobrepõe o valor derivado de `altitude_km`;
        serve pra re-resolver com o n já corrigido pela calibração de
        Kozai.
        """
        if revolutions < 0:
            raise ValueError(f"revolutions não pode ser negativo: {revolutions}")
        if epoch.tzinfo is None or epoch.utcoffset() is None:
            raise ValueError("epoch precisa ser timezone-aware")

        plane_target = plane_target or OverheadTarget()
        mean_motion = mean_motion_rad_s or mean_motion_from_altitude(altitude_km)
        period_s = _TWO_PI / mean_motion

        u_origin = self._earth.to_eci_unit(origin, self._sidereal_time.gmst_rad(epoch))

        time_of_flight = 0.0
        for iteration in range(1, self._max_iter + 1):
            arrival = epoch + dt.timedelta(seconds=time_of_flight)
            u_target = self._earth.to_eci_unit(
                target, self._sidereal_time.gmst_rad(arrival)
            )
            h_hat = plane_target.constrain(u_origin, u_target)

            # Projetar é no-op para OverheadTarget e essencial para
            # MaxElevationTarget, onde o alvo fica fora do plano.
            u_origin_ang = argument_of_latitude(
                h_hat, project_onto_plane(u_origin, h_hat)
            )
            u_target_ang = argument_of_latitude(
                h_hat, project_onto_plane(u_target, h_hat)
            )
            arc = (u_target_ang - u_origin_ang) % _TWO_PI

            predicted = (arc / _TWO_PI + revolutions) * period_s
            residual = predicted - time_of_flight
            time_of_flight += self._relaxation * residual

            if abs(residual) < self._tol_s:
                inc_rad, raan_rad = inclination_raan(h_hat)
                elements = OrbitalElements(
                    inc_rad=inc_rad,
                    raan_rad=raan_rad,
                    ecc=CIRCULAR_ECC,
                    argp_rad=0.0,
                    mean_anomaly_rad=u_origin_ang,
                    mean_motion_rad_s=mean_motion,
                )
                return PassSolution(
                    elements=elements,
                    epoch=epoch,
                    time_of_flight_s=time_of_flight,
                    iterations=iteration,
                )

        raise ConvergenceError(
            f"sem convergência em {self._max_iter} iterações "
            f"(resíduo {residual:.3f} s); tente relaxation < 1.0"
        )
