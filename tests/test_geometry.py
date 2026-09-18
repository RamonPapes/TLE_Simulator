"""Geometria pura: casos conferíveis no papel, com a Terra parada."""

from __future__ import annotations

import math

import numpy as np
import pytest

from my_pass_prediction.fake_tle import GroundPoint, SphericalEarth
from my_pass_prediction.fake_tle.geometry import (
    DegenerateGeometryError,
    argument_of_latitude,
    ascending_node_vector,
    central_angle_for_elevation,
    inclination_raan,
    mean_motion_from_altitude,
    plane_from_directions,
    project_onto_plane,
    rotate_about_axis,
    semi_major_axis_from_mean_motion,
)

EARTH = SphericalEarth()


def eci(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Direção ECI com GMST = 0: a longitude vira ascensão reta direta."""
    return EARTH.to_eci_unit(GroundPoint(lat_deg, lon_deg), 0.0)


class TestMeanMotion:
    def test_leo_500km(self):
        assert mean_motion_from_altitude(500.0) == pytest.approx(1.1068e-3, rel=1e-3)

    def test_periodo_de_500km_e_95_minutos(self):
        period_min = 2 * math.pi / mean_motion_from_altitude(500.0) / 60.0
        assert period_min == pytest.approx(94.6, abs=0.1)

    def test_rev_por_dia_de_500km(self):
        rev_day = 86400.0 / (2 * math.pi / mean_motion_from_altitude(500.0))
        assert rev_day == pytest.approx(15.219, abs=0.001)

    def test_e_inverso_de_semi_major_axis(self):
        mean_motion = mean_motion_from_altitude(500.0)
        assert semi_major_axis_from_mean_motion(mean_motion) == pytest.approx(
            6878.137, abs=1e-6
        )


class TestPlaneFromDirections:
    def test_dois_pontos_no_equador_dao_orbita_equatorial(self):
        inc, _ = inclination_raan(plane_from_directions(eci(0, 0), eci(0, 90)))
        assert math.degrees(inc) == pytest.approx(0.0, abs=1e-9)

    def test_mesmo_meridiano_da_orbita_polar(self):
        h_hat = plane_from_directions(eci(10, 0), eci(50, 0))
        inc, raan = inclination_raan(h_hat)
        assert math.degrees(inc) == pytest.approx(90.0, abs=1e-9)
        assert math.degrees(raan) == pytest.approx(0.0, abs=1e-9)

    def test_momento_angular_conhecido(self):
        h_hat = plane_from_directions(eci(0, 0), eci(0, 90))
        np.testing.assert_allclose(h_hat, [0.0, 0.0, 1.0], atol=1e-12)

    def test_retrogrado_inverte_o_sinal(self):
        u_a, u_b = eci(-10, 20), eci(30, 70)
        np.testing.assert_allclose(
            plane_from_directions(u_a, u_b, retrograde=True),
            -plane_from_directions(u_a, u_b),
            atol=1e-12,
        )

    def test_pontos_coincidentes_sao_degenerados(self):
        with pytest.raises(DegenerateGeometryError):
            plane_from_directions(eci(10, 20), eci(10, 20))

    def test_pontos_antipodas_sao_degenerados(self):
        with pytest.raises(DegenerateGeometryError):
            plane_from_directions(eci(10, 20), eci(-10, -160))


class TestInvariantes:
    """Valem para qualquer par de pontos, não só os escolhidos a dedo."""

    PARES = [
        (0.0, 0.0, 0.0, 90.0),
        (-12.98, -38.50, -23.55, -46.63),
        (45.0, 10.0, -45.0, 170.0),
        (80.0, -120.0, -5.0, 60.0),
        (-33.9, 151.2, 51.5, -0.1),
    ]

    @pytest.mark.parametrize("lat_a,lon_a,lat_b,lon_b", PARES)
    def test_plano_contem_os_dois_pontos(self, lat_a, lon_a, lat_b, lon_b):
        u_a, u_b = eci(lat_a, lon_a), eci(lat_b, lon_b)
        h_hat = plane_from_directions(u_a, u_b)
        assert float(np.dot(h_hat, u_a)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.dot(h_hat, u_b)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.linalg.norm(h_hat)) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("lat_a,lon_a,lat_b,lon_b", PARES)
    def test_inclinacao_cobre_as_duas_latitudes(self, lat_a, lon_a, lat_b, lon_b):
        """Uma órbita só alcança latitudes até a própria inclinação."""
        inc, _ = inclination_raan(plane_from_directions(eci(lat_a, lon_a), eci(lat_b, lon_b)))
        reach_deg = math.degrees(min(inc, math.pi - inc))
        assert reach_deg >= max(abs(lat_a), abs(lat_b)) - 1e-9

    @pytest.mark.parametrize("lat_a,lon_a,lat_b,lon_b", PARES)
    def test_no_ascendente_esta_no_plano_e_no_equador(self, lat_a, lon_a, lat_b, lon_b):
        h_hat = plane_from_directions(eci(lat_a, lon_a), eci(lat_b, lon_b))
        node = ascending_node_vector(h_hat)
        assert float(np.dot(h_hat, node)) == pytest.approx(0.0, abs=1e-12)
        assert float(node[2]) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("lat_a,lon_a,lat_b,lon_b", PARES)
    def test_arcos_direto_e_retrogrado_somam_uma_volta(self, lat_a, lon_a, lat_b, lon_b):
        u_a, u_b = eci(lat_a, lon_a), eci(lat_b, lon_b)
        arcs = []
        for retrograde in (False, True):
            h_hat = plane_from_directions(u_a, u_b, retrograde)
            arc = (
                argument_of_latitude(h_hat, u_b) - argument_of_latitude(h_hat, u_a)
            ) % (2 * math.pi)
            arcs.append(arc)
        assert math.degrees(sum(arcs)) == pytest.approx(360.0, abs=1e-9)


class TestArgumentOfLatitude:
    def test_no_ascendente_tem_argumento_zero(self):
        h_hat = plane_from_directions(eci(10, 0), eci(50, 0))
        node = ascending_node_vector(h_hat)
        assert argument_of_latitude(h_hat, node) == pytest.approx(0.0, abs=1e-12)

    def test_um_quarto_de_volta(self):
        h_hat = plane_from_directions(eci(0, 0), eci(0, 90))
        assert math.degrees(argument_of_latitude(h_hat, eci(0, 90))) == pytest.approx(
            90.0, abs=1e-9
        )

    def test_cresce_no_sentido_do_movimento(self):
        h_hat = plane_from_directions(eci(0, 0), eci(0, 90))
        sequence = [
            math.degrees(argument_of_latitude(h_hat, eci(0, lon)))
            for lon in (0, 45, 90, 135)
        ]
        assert sequence == sorted(sequence)


class TestProjectOntoPlane:
    def test_vetor_no_plano_nao_muda(self):
        h_hat = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            project_onto_plane([1.0, 0.0, 0.0], h_hat), [1.0, 0.0, 0.0], atol=1e-12
        )

    def test_remove_a_componente_normal(self):
        h_hat = np.array([0.0, 0.0, 1.0])
        projected = project_onto_plane([1.0, 0.0, 1.0], h_hat)
        assert float(np.dot(projected, h_hat)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.linalg.norm(projected)) == pytest.approx(1.0, abs=1e-12)


class TestRotateAboutAxis:
    def test_quarto_de_volta_em_z(self):
        np.testing.assert_allclose(
            rotate_about_axis([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], math.pi / 2),
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )

    def test_eixo_e_invariante(self):
        axis = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            rotate_about_axis(axis, axis, 1.234), axis, atol=1e-12
        )

    def test_preserva_a_norma(self):
        vec = np.array([0.3, -0.5, 0.81])
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
        rotated = rotate_about_axis(vec, axis, 0.7)
        assert float(np.linalg.norm(rotated)) == pytest.approx(
            float(np.linalg.norm(vec)), abs=1e-12
        )


class TestCentralAngleForElevation:
    def test_zenite_nao_tem_desvio(self):
        assert central_angle_for_elevation(90.0, 500.0) == pytest.approx(0.0, abs=1e-12)

    def test_horizonte_em_500km(self):
        gamma = math.degrees(central_angle_for_elevation(0.0, 500.0))
        assert gamma == pytest.approx(21.99, abs=0.01)

    def test_monotonica_decrescente_na_elevacao(self):
        gammas = [central_angle_for_elevation(e, 500.0) for e in (0, 15, 30, 60, 90)]
        assert gammas == sorted(gammas, reverse=True)

    def test_recusa_elevacao_fora_de_faixa(self):
        with pytest.raises(ValueError):
            central_angle_for_elevation(95.0, 500.0)
