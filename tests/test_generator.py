"""Integração: o TLE gerado, propagado pelo SGP4 de verdade.

Nenhum teste aqui usa rede -- o TLE é construído localmente.

Sobre as tolerâncias: o resíduo de ~0.1 grau não vem do modelo de Terra,
vem do J2. Os elementos do TLE são médios (brouwerianos) e o SGP4 devolve
posição osculante; a diferença entre as duas é da ordem de um décimo de
grau no solo e não some apertando o elipsoide.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
from passpredict import Location, Observer, SGP4Propagator

from my_pass_prediction.fake_tle import (
    ConvergenceError,
    FakeTleGenerator,
    GroundPoint,
    Iau82SiderealTime,
    KozaiCalibrator,
    MaxElevationTarget,
    OrbitalElements,
    OverheadPassSolver,
    PatchedTLE,
    Sgp4AltitudeProbe,
    TleIdentity,
    Wgs84Earth,
    generate_fake_tle,
    subpoint,
)
from my_pass_prediction.fake_tle.geometry import (
    R_EARTH_KM,
    mean_motion_from_altitude,
    semi_major_axis_from_mean_motion,
)

POSITION_TOL_DEG = 0.15
ALTITUDE_TOL_KM = 0.5


def separation_deg(lat_a, lon_a, lat_b, lon_b) -> float:
    """Separação angular entre dois pontos no solo, em graus."""
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    delta = math.radians(lon_b - lon_a)
    cosine = math.sin(phi_a) * math.sin(phi_b) + math.cos(phi_a) * math.cos(
        phi_b
    ) * math.cos(delta)
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


class FakeProbe:
    """AltitudeProbe que reporta a altitude kepleriana com um viés fixo."""

    def __init__(self, bias_km: float) -> None:
        self.bias_km = bias_km
        self.calls = 0

    def mean_altitude_km(self, elements: OrbitalElements, epoch: dt.datetime) -> float:
        self.calls += 1
        keplerian = (
            semi_major_axis_from_mean_motion(elements.mean_motion_rad_s) - R_EARTH_KM
        )
        return keplerian + self.bias_km


class TestKozaiCalibrator:
    """Unitário: a calibração não precisa do SGP4, só do protocol."""

    def test_remove_o_vies_da_sonda(self):
        probe = FakeProbe(bias_km=-3.0)
        elements = OrbitalElements(
            1.0, 0.0, 1e-7, 0.0, 0.0, mean_motion_from_altitude(500.0)
        )
        calibrated = KozaiCalibrator(probe, tol_km=1e-4).calibrate(
            elements, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), 500.0
        )
        corrected_altitude = (
            semi_major_axis_from_mean_motion(calibrated.mean_motion_rad_s) - R_EARTH_KM
        )
        assert corrected_altitude == pytest.approx(503.0, abs=1e-3)

    def test_nao_faz_nada_quando_ja_esta_certo(self):
        probe = FakeProbe(bias_km=0.0)
        elements = OrbitalElements(
            1.0, 0.0, 1e-7, 0.0, 0.0, mean_motion_from_altitude(500.0)
        )
        calibrated = KozaiCalibrator(probe).calibrate(
            elements, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), 500.0
        )
        assert calibrated.mean_motion_rad_s == elements.mean_motion_rad_s
        assert probe.calls == 1

    def test_levanta_se_nao_convergir(self):
        class HopelessProbe:
            def mean_altitude_km(self, elements, epoch):
                return 0.0  # nunca se aproxima do alvo

        elements = OrbitalElements(
            1.0, 0.0, 1e-7, 0.0, 0.0, mean_motion_from_altitude(500.0)
        )
        with pytest.raises(ConvergenceError, match="Kozai"):
            KozaiCalibrator(HopelessProbe(), max_iter=3).calibrate(
                elements, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), 500.0
            )


@pytest.mark.integration
class TestPassagemPeloZenite:
    @pytest.fixture
    def generated(self, salvador, sao_paulo, epoch):
        tle, solution = generate_fake_tle(salvador, sao_paulo, epoch)
        return tle, solution, SGP4Propagator.from_tle(tle)

    def test_satelite_esta_sobre_a_origem_na_epoca(self, generated, salvador, epoch):
        _, _, satellite = generated
        lat, lon, _ = subpoint(satellite, epoch)
        assert separation_deg(lat, lon, salvador.lat_deg, salvador.lon_deg) < (
            POSITION_TOL_DEG
        )

    def test_satelite_esta_sobre_o_alvo_na_chegada(self, generated, sao_paulo):
        _, solution, satellite = generated
        lat, lon, _ = subpoint(satellite, solution.arrival)
        assert separation_deg(lat, lon, sao_paulo.lat_deg, sao_paulo.lon_deg) < (
            POSITION_TOL_DEG
        )

    def test_altitude_media_e_a_pedida(self, generated, epoch):
        _, solution, satellite = generated
        step = solution.elements.period_s / 64
        radii = [
            float(
                np.linalg.norm(
                    satellite.get_only_position(epoch + dt.timedelta(seconds=i * step))
                )
            )
            for i in range(64)
        ]
        assert float(np.mean(radii)) - R_EARTH_KM == pytest.approx(
            500.0, abs=ALTITUDE_TOL_KM
        )

    def test_elevacao_no_alvo_e_praticamente_zenital(self, generated, sao_paulo):
        """Fecha o ciclo com o mesmo Observer que o resto do projeto usa."""
        _, solution, satellite = generated
        observer = Observer(
            Location("alvo", sao_paulo.lat_deg, sao_paulo.lon_deg, 0), satellite
        )
        assert observer.elevation(solution.arrival) > 88.0

    def test_tempo_de_voo_e_menos_de_uma_orbita(self, generated):
        _, solution, _ = generated
        assert 0.0 < solution.revolutions < 1.0


@pytest.mark.integration
class TestOpcoes:
    def test_revolutions_adia_a_chegada(self, salvador, sao_paulo, epoch):
        _, first = generate_fake_tle(salvador, sao_paulo, epoch, revolutions=0)
        _, later = generate_fake_tle(salvador, sao_paulo, epoch, revolutions=2)
        assert 1.9 < later.revolutions - first.revolutions < 2.1

    def test_sem_calibracao_a_altitude_erra_pouco(self, salvador, sao_paulo, epoch):
        _, solution = generate_fake_tle(salvador, sao_paulo, epoch, calibrate=False)
        assert solution.elements.mean_motion_rad_s == pytest.approx(
            mean_motion_from_altitude(500.0), rel=1e-12
        )

    def test_calibracao_muda_o_mean_motion(self, salvador, sao_paulo, epoch):
        _, raw = generate_fake_tle(salvador, sao_paulo, epoch, calibrate=False)
        _, tuned = generate_fake_tle(salvador, sao_paulo, epoch, calibrate=True)
        assert tuned.elements.mean_motion_rad_s != raw.elements.mean_motion_rad_s
        # Kozai é uma correção pequena: décimos de km, não dezenas.
        assert tuned.elements.mean_motion_rad_s == pytest.approx(
            raw.elements.mean_motion_rad_s, rel=1e-3
        )

    def test_altitude_diferente_muda_o_periodo(self, salvador, sao_paulo, epoch):
        _, low = generate_fake_tle(salvador, sao_paulo, epoch, altitude_km=400.0)
        _, high = generate_fake_tle(salvador, sao_paulo, epoch, altitude_km=800.0)
        assert low.elements.period_s < high.elements.period_s

    def test_elevacao_maxima_pedida_e_atendida(self, epoch):
        """Alvo distante o suficiente para comportar o desvio."""
        origin, target = GroundPoint(-12.9777, -38.5016), GroundPoint(-30.0, -60.0)
        tle, solution = generate_fake_tle(
            origin, target, epoch, plane_target=MaxElevationTarget(45.0)
        )
        observer = Observer(
            Location("alvo", target.lat_deg, target.lon_deg, 0),
            SGP4Propagator.from_tle(tle),
        )
        assert observer.elevation(solution.arrival) == pytest.approx(45.0, abs=2.0)

    def test_identity_aparece_no_tle(self, salvador, sao_paulo, epoch):
        identity = TleIdentity(satnum=12345, name="QKD-SAT")
        tle, _ = generate_fake_tle(salvador, sao_paulo, epoch, identity=identity)
        assert tle.satid == 12345
        assert tle.name == "QKD-SAT"
        assert tle.tle1[2:7] == "12345"


@pytest.mark.integration
class TestRegressaoPasspredict:
    """O passpredict 0.5.1 lê a coluna de inclinação deslocada em 1."""

    def test_patched_tle_le_inclinacao_de_tres_digitos(self, iss_lines):
        line2 = "2 99999 124.1083 327.8393 0000001   0.0000 195.6347 15.219592500000"
        assert PatchedTLE(99999, (iss_lines[0], line2)).inc == pytest.approx(124.1083)

    def test_patched_tle_concorda_com_o_original_abaixo_de_100_graus(self, iss_lines):
        """Onde o bug não morde, o comportamento tem que ser idêntico."""
        from passpredict import TLE

        patched = PatchedTLE(25544, iss_lines)
        assert patched.inc == pytest.approx(TLE(25544, iss_lines).inc)
        assert patched.inc == pytest.approx(51.6307)

    def test_orbita_retrograda_propaga_na_inclinacao_certa(self, epoch):
        """Sem o patch, uma órbita de 124 graus seria propagada como 24."""
        origin, target = GroundPoint(-12.9777, -38.5016), GroundPoint(-23.5505, -46.6333)
        tle, solution = generate_fake_tle(origin, target, epoch)
        assert math.degrees(solution.elements.inc_rad) > 100.0

        satellite = SGP4Propagator.from_tle(tle)
        lat, lon, _ = subpoint(satellite, epoch)
        assert separation_deg(lat, lon, origin.lat_deg, origin.lon_deg) < (
            POSITION_TOL_DEG
        )


@pytest.mark.integration
class TestInjecaoDeDependencia:
    def test_generator_montado_na_mao(self, salvador, sao_paulo, epoch):
        """O caminho SOLID: nada de padrão escondido."""
        identity = TleIdentity(satnum=777, name="INJETADO")
        solver = OverheadPassSolver(Iau82SiderealTime(), Wgs84Earth())
        generator = FakeTleGenerator(
            solver, identity, KozaiCalibrator(Sgp4AltitudeProbe(identity))
        )
        tle, solution = generator.generate(salvador, sao_paulo, epoch)
        assert tle.satid == 777
        satellite = SGP4Propagator.from_tle(tle)
        lat, lon, _ = subpoint(satellite, solution.arrival)
        assert separation_deg(lat, lon, sao_paulo.lat_deg, sao_paulo.lon_deg) < (
            POSITION_TOL_DEG
        )

    def test_solver_sem_calibrador(self, salvador, sao_paulo, epoch):
        generator = FakeTleGenerator(
            OverheadPassSolver(Iau82SiderealTime(), Wgs84Earth())
        )
        tle, solution = generator.generate(salvador, sao_paulo, epoch)
        assert solution.elements.mean_motion_rad_s == pytest.approx(
            mean_motion_from_altitude(500.0), rel=1e-12
        )
