"""O laço de ponto fixo.

Com a Terra parada o tempo de voo fecha analiticamente: não há iteração
a fazer, então a asserção é exata. Com o GMST real o que se testa é a
convergência.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from my_pass_prediction.fake_tle import (
    ConvergenceError,
    DegenerateGeometryError,
    GroundPoint,
    MaxElevationTarget,
    OverheadPassSolver,
    OverheadTarget,
)
from my_pass_prediction.fake_tle.geometry import mean_motion_from_altitude
from my_pass_prediction.fake_tle.solver import CIRCULAR_ECC

PERIOD_500KM_S = 2 * math.pi / mean_motion_from_altitude(500.0)


class TestTerraParada:
    """GMST fixo em 0: a longitude é ascensão reta e o arco é conhecido."""

    def test_quarto_de_volta_leva_um_quarto_de_periodo(self, frozen_solver, epoch):
        solution = frozen_solver.solve(GroundPoint(0, 0), GroundPoint(0, 90), epoch)
        assert solution.time_of_flight_s == pytest.approx(PERIOD_500KM_S / 4, abs=1e-6)

    def test_arco_curto_vale_para_qualquer_sentido(self, frozen_solver, epoch):
        """O plano é orientado para o satélite ir de A a B pelo arco curto,
        então ir para leste ou para oeste custa o mesmo tempo."""
        east = frozen_solver.solve(GroundPoint(0, 0), GroundPoint(0, 90), epoch)
        west = frozen_solver.solve(GroundPoint(0, 0), GroundPoint(0, -90), epoch)
        assert west.time_of_flight_s == pytest.approx(
            east.time_of_flight_s, abs=1e-9
        )
        assert west.time_of_flight_s == pytest.approx(PERIOD_500KM_S / 4, abs=1e-6)

    def test_tres_quartos_de_volta_pelo_caminho_longo(self, frozen_solver, epoch):
        solution = frozen_solver.solve(
            GroundPoint(0, 0),
            GroundPoint(0, 90),
            epoch,
            plane_target=OverheadTarget(retrograde=True),
        )
        assert solution.time_of_flight_s == pytest.approx(
            3 * PERIOD_500KM_S / 4, abs=1e-6
        )

    def test_converge_em_duas_iteracoes(self, frozen_solver, epoch):
        """Sem rotação não há o que iterar: uma passada calcula o tempo de
        voo e a segunda confirma que ele não mudou."""
        solution = frozen_solver.solve(GroundPoint(0, 0), GroundPoint(0, 90), epoch)
        assert solution.iterations == 2

    def test_revolutions_soma_periodos_inteiros(self, frozen_solver, epoch):
        base = frozen_solver.solve(GroundPoint(0, 0), GroundPoint(0, 90), epoch)
        for revolutions in (1, 3):
            later = frozen_solver.solve(
                GroundPoint(0, 0), GroundPoint(0, 90), epoch, revolutions=revolutions
            )
            assert later.time_of_flight_s == pytest.approx(
                base.time_of_flight_s + revolutions * PERIOD_500KM_S, abs=1e-6
            )

    def test_retrogrado_percorre_o_arco_complementar(self, frozen_solver, epoch):
        direct = frozen_solver.solve(GroundPoint(10, 0), GroundPoint(40, 30), epoch)
        retro = frozen_solver.solve(
            GroundPoint(10, 0),
            GroundPoint(40, 30),
            epoch,
            plane_target=OverheadTarget(retrograde=True),
        )
        assert (
            direct.time_of_flight_s + retro.time_of_flight_s
        ) == pytest.approx(PERIOD_500KM_S, abs=1e-6)


class TestElementosResultantes:
    def test_orbita_e_circular(self, frozen_solver, epoch):
        elements = frozen_solver.solve(
            GroundPoint(0, 0), GroundPoint(0, 90), epoch
        ).elements
        assert elements.ecc == CIRCULAR_ECC
        assert elements.argp_rad == 0.0

    def test_mean_motion_vem_da_altitude(self, frozen_solver, epoch):
        for altitude_km in (400.0, 500.0, 800.0):
            elements = frozen_solver.solve(
                GroundPoint(0, 0), GroundPoint(0, 90), epoch, altitude_km=altitude_km
            ).elements
            assert elements.mean_motion_rad_s == pytest.approx(
                mean_motion_from_altitude(altitude_km), rel=1e-12
            )

    def test_mean_motion_explicito_sobrepoe_a_altitude(self, frozen_solver, epoch):
        override = 1.1e-3
        elements = frozen_solver.solve(
            GroundPoint(0, 0),
            GroundPoint(0, 90),
            epoch,
            mean_motion_rad_s=override,
        ).elements
        assert elements.mean_motion_rad_s == override

    def test_anomalia_media_posiciona_o_satelite_sobre_a_origem(
        self, frozen_solver, spherical_earth, epoch
    ):
        """Reconstrói a posição pelos elementos e compara com a origem."""
        origin = GroundPoint(-12.98, -38.50)
        elements = frozen_solver.solve(origin, GroundPoint(-23.55, -46.63), epoch).elements
        inc, raan, u = elements.inc_rad, elements.raan_rad, elements.mean_anomaly_rad
        reconstructed = np.array(
            [
                math.cos(u) * math.cos(raan) - math.sin(u) * math.cos(inc) * math.sin(raan),
                math.cos(u) * math.sin(raan) + math.sin(u) * math.cos(inc) * math.cos(raan),
                math.sin(u) * math.sin(inc),
            ]
        )
        np.testing.assert_allclose(
            reconstructed, spherical_earth.to_eci_unit(origin, 0.0), atol=1e-12
        )


class TestTerraGirando:
    def test_converge_rapido(self, real_solver, salvador, sao_paulo, epoch):
        solution = real_solver.solve(salvador, sao_paulo, epoch)
        assert solution.iterations < 20

    def test_o_ponto_fixo_e_consistente(self, real_solver, salvador, sao_paulo, epoch):
        """Re-resolver a partir do resultado tem que dar o mesmo tempo."""
        first = real_solver.solve(salvador, sao_paulo, epoch)
        second = real_solver.solve(salvador, sao_paulo, epoch)
        assert first.time_of_flight_s == pytest.approx(second.time_of_flight_s, abs=1e-9)

    def test_rotacao_da_terra_muda_o_resultado(
        self, real_solver, frozen_solver, salvador, sao_paulo, epoch
    ):
        """Se isto falhar, o GMST não está sendo usado."""
        turning = real_solver.solve(salvador, sao_paulo, epoch)
        frozen = frozen_solver.solve(salvador, sao_paulo, epoch)
        assert turning.time_of_flight_s != pytest.approx(
            frozen.time_of_flight_s, abs=1e-3
        )

    def test_arrival_e_epoch_mais_tempo_de_voo(self, real_solver, salvador, sao_paulo, epoch):
        solution = real_solver.solve(salvador, sao_paulo, epoch)
        assert solution.arrival == epoch + dt.timedelta(
            seconds=solution.time_of_flight_s
        )


class TestFalhas:
    def test_sem_convergencia_levanta(self, real_sidereal, wgs84_earth, salvador, sao_paulo, epoch):
        solver = OverheadPassSolver(real_sidereal, wgs84_earth, max_iter=1)
        with pytest.raises(ConvergenceError, match="convergência"):
            solver.solve(salvador, sao_paulo, epoch)

    def test_epoch_ingenuo_e_recusado(self, frozen_solver, salvador, sao_paulo):
        with pytest.raises(ValueError, match="timezone-aware"):
            frozen_solver.solve(salvador, sao_paulo, dt.datetime(2026, 9, 18, 12))

    def test_revolucoes_negativas_sao_recusadas(self, frozen_solver, salvador, sao_paulo, epoch):
        with pytest.raises(ValueError, match="revolutions"):
            frozen_solver.solve(salvador, sao_paulo, epoch, revolutions=-1)

    def test_pontos_iguais_sao_degenerados(self, frozen_solver, salvador, epoch):
        with pytest.raises(DegenerateGeometryError):
            frozen_solver.solve(salvador, salvador, epoch)

    def test_relaxation_invalido(self, real_sidereal, wgs84_earth):
        with pytest.raises(ValueError, match="relaxation"):
            OverheadPassSolver(real_sidereal, wgs84_earth, relaxation=0.0)


class TestMaxElevationTarget:
    @pytest.mark.parametrize("elevation_deg", [30.0, 60.0, 85.0])
    def test_origem_continua_sob_a_ground_track(
        self, frozen_solver, spherical_earth, epoch, elevation_deg
    ):
        """A rotação em torno de u_a preserva a passagem pelo zênite da origem."""
        origin, target = GroundPoint(10, 0), GroundPoint(40, 30)
        solution = frozen_solver.solve(
            origin,
            target,
            epoch,
            plane_target=MaxElevationTarget(elevation_deg),
        )
        inc, raan = solution.elements.inc_rad, solution.elements.raan_rad
        h_hat = np.array(
            [math.sin(inc) * math.sin(raan), -math.sin(inc) * math.cos(raan), math.cos(inc)]
        )
        u_origin = spherical_earth.to_eci_unit(origin, 0.0)
        assert float(np.dot(h_hat, u_origin)) == pytest.approx(0.0, abs=1e-9)

    def test_elevacao_de_90_graus_equivale_ao_zenite(self, frozen_solver, epoch):
        origin, target = GroundPoint(10, 0), GroundPoint(40, 30)
        overhead = frozen_solver.solve(origin, target, epoch)
        via_elevation = frozen_solver.solve(
            origin, target, epoch, plane_target=MaxElevationTarget(90.0)
        )
        assert via_elevation.elements.inc_rad == pytest.approx(
            overhead.elements.inc_rad, abs=1e-9
        )

    def test_desvio_maior_que_a_separacao_e_impossivel(self, frozen_solver, epoch):
        """Pontos quase juntos não comportam uma elevação baixa no alvo."""
        with pytest.raises(DegenerateGeometryError, match="separação angular"):
            frozen_solver.solve(
                GroundPoint(0, 0),
                GroundPoint(0, 1),
                epoch,
                plane_target=MaxElevationTarget(0.0),
            )

    def test_offset_sign_invalido(self):
        with pytest.raises(ValueError, match="offset_sign"):
            MaxElevationTarget(45.0, offset_sign=0)
