"""Amostragem da trajetória.

A maior parte dos testes usa dublês no lugar do SGP4: `sample_track` só
precisa de algo que saiba dar posição e razel, e stubs deixam as
asserções exatas -- além de provarem quantas propagações aconteceram,
que é o ponto do teto de amostras.
"""

from __future__ import annotations

import datetime as dt

import pytest

from my_pass_prediction.fake_tle import GroundPoint, generate_fake_tle, point_at_bearing
from my_pass_prediction.web.track import (
    NoPassFoundError,
    PassTrack,
    TooManySamplesError,
    TrackSample,
    build_pass_track,
    sample_track,
)

UTC = dt.timezone.utc


class FakeSatellite:
    """Fica parado sobre o equador, e conta quantas vezes foi propagado."""

    def __init__(self) -> None:
        self.calls = 0

    def get_only_position(self, when: dt.datetime):
        self.calls += 1
        return (6878.137, 0.0, 0.0)  # 500 km acima de (0, 0)


class FakeObserver:
    """Devolve sempre o mesmo razel, e conta as chamadas."""

    def __init__(self, el: float = 45.0) -> None:
        self.calls = 0
        self._el = el

    def razel(self, when: dt.datetime):
        self.calls += 1
        return type("RazEl", (), {"range": 700.0, "az": 90.0, "el": self._el})()


@pytest.fixture
def fakes():
    return FakeSatellite(), FakeObserver()


@pytest.fixture
def window(epoch):
    return epoch, epoch + dt.timedelta(seconds=100)


class TestSampleTrack:
    def test_conta_amostras_de_janela_exata(self, fakes, epoch, window):
        satellite, observer = fakes
        samples = sample_track(satellite, observer, epoch, *window, step_s=10.0)
        assert len(samples) == 11  # 0, 10, ..., 100

    def test_inclui_o_instante_final_em_janela_nao_multipla(self, fakes, epoch):
        satellite, observer = fakes
        samples = sample_track(
            satellite,
            observer,
            epoch,
            epoch,
            epoch + dt.timedelta(seconds=95),
            step_s=10.0,
        )
        assert samples[-1].offset_s == pytest.approx(95.0)
        assert len(samples) == 11  # 0..90 mais o 95 final

    def test_offset_e_relativo_a_epoca_e_pode_ser_negativo(self, fakes, epoch):
        satellite, observer = fakes
        samples = sample_track(
            satellite,
            observer,
            epoch,
            epoch - dt.timedelta(seconds=30),
            epoch + dt.timedelta(seconds=30),
            step_s=30.0,
        )
        assert [s.offset_s for s in samples] == pytest.approx([-30.0, 0.0, 30.0])

    def test_subponto_do_dublê_cai_na_origem(self, fakes, epoch, window):
        satellite, observer = fakes
        sample = sample_track(satellite, observer, epoch, *window, step_s=100.0)[0]
        assert sample.lat_deg == pytest.approx(0.0, abs=1e-6)
        assert sample.lon_deg == pytest.approx(0.0, abs=1e-6)
        assert sample.alt_km == pytest.approx(500.0, abs=0.5)

    def test_step_nao_positivo_levanta(self, fakes, epoch, window):
        satellite, observer = fakes
        with pytest.raises(ValueError, match="step_s"):
            sample_track(satellite, observer, epoch, *window, step_s=0.0)

    def test_janela_invertida_levanta(self, fakes, epoch):
        satellite, observer = fakes
        with pytest.raises(ValueError, match="janela vazia"):
            sample_track(
                satellite,
                observer,
                epoch,
                epoch,
                epoch - dt.timedelta(seconds=10),
                step_s=1.0,
            )

    def test_teto_de_amostras_recusa_antes_de_propagar(self, fakes, epoch, window):
        """O ponto do teto: recusar de graça, não depois de pagar a conta."""
        satellite, observer = fakes
        with pytest.raises(TooManySamplesError, match="excedem o teto"):
            sample_track(satellite, observer, epoch, *window, step_s=1.0, max_samples=10)
        assert satellite.calls == 0
        assert observer.calls == 0

    def test_teto_folgado_nao_atrapalha(self, fakes, epoch, window):
        satellite, observer = fakes
        samples = sample_track(
            satellite, observer, epoch, *window, step_s=10.0, max_samples=11
        )
        assert len(samples) == 11
        assert satellite.calls == 11


class TestPassTrack:
    def _sample(self, offset: float, el: float, epoch: dt.datetime) -> TrackSample:
        return TrackSample(
            offset_s=offset,
            when=epoch + dt.timedelta(seconds=offset),
            lat_deg=0.0,
            lon_deg=0.0,
            alt_km=500.0,
            elevation_deg=el,
            azimuth_deg=0.0,
            range_km=700.0,
        )

    def test_trajetoria_vazia_levanta(self, salvador, epoch):
        with pytest.raises(ValueError, match="pelo menos uma amostra"):
            PassTrack(salvador, "X", epoch, (), None, None, None)

    def test_duracao_e_elevacao_maxima(self, salvador, epoch):
        samples = tuple(
            self._sample(offset, el, epoch)
            for offset, el in ((-60.0, -3.0), (0.0, 80.0), (60.0, 5.0))
        )
        track = PassTrack(salvador, "X", epoch, samples, None, None, None)
        assert track.duration_s == pytest.approx(120.0)
        assert track.max_elevation_deg == pytest.approx(80.0)

    def test_sem_marcadores_a_duracao_da_passagem_e_desconhecida(self, salvador, epoch):
        track = PassTrack(
            salvador, "X", epoch, (self._sample(0.0, 10.0, epoch),), None, None, None
        )
        assert track.pass_duration_s is None

    def test_visibilidade_segue_o_sinal_da_elevacao(self, epoch):
        assert self._sample(0.0, 0.1, epoch).is_visible
        assert not self._sample(0.0, -0.1, epoch).is_visible


@pytest.mark.integration
class TestBuildPassTrack:
    @pytest.fixture
    def satellite(self, salvador, epoch):
        from passpredict import SGP4Propagator

        heading = GroundPoint(
            *point_at_bearing(salvador.lat_deg, salvador.lon_deg, 200.0, 20.0)
        )
        tle, _ = generate_fake_tle(salvador, heading, epoch, altitude_km=500.0)
        return SGP4Propagator.from_tle(tle)

    def test_epoca_ingenua_levanta(self, satellite, salvador):
        with pytest.raises(ValueError, match="timezone-aware"):
            build_pass_track(satellite, salvador, dt.datetime(2026, 9, 18, 12, 0))

    def test_a_passagem_encontrada_e_a_centrada_na_epoca(
        self, satellite, salvador, epoch
    ):
        """O TLE sintético põe o zênite na época; o TCA tem que bater."""
        track = build_pass_track(satellite, salvador, epoch, station_name="Salvador")
        assert abs((track.tca.when - epoch).total_seconds()) < 30.0
        assert track.tca.elevation_deg > 85.0

    def test_janela_cobre_aos_ate_los_com_folga(self, satellite, salvador, epoch):
        pad = 120.0
        track = build_pass_track(satellite, salvador, epoch, pad_s=pad, step_s=10.0)
        assert track.samples[0].when == pytest.approx(
            track.aos.when - dt.timedelta(seconds=pad), abs=dt.timedelta(seconds=1)
        )
        assert track.samples[0].elevation_deg < 0.0
        assert track.samples[-1].elevation_deg < 0.0

    def test_estacao_antipoda_nao_ve_a_passagem(self, satellite, epoch):
        antipoda = GroundPoint(12.9777, 141.4984)
        with pytest.raises(NoPassFoundError, match="nenhuma passagem"):
            build_pass_track(satellite, antipoda, epoch, search_back_s=600.0)
