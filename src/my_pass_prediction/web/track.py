"""Amostragem de uma passagem: da órbita para a série temporal do mapa.

Camada de dados da interface web. Recebe um satélite já propagável e uma
estação em solo, devolve a trajetória amostrada. Não sabe o que é HTTP,
JSON ou Leaflet -- a serialização mora em `api.py`, e quem desenha é o
navegador.

A janela padrão é a própria passagem (AOS a LOS, com uma folga), e não
um intervalo fixo em volta da época: uma passagem de 500 km dura uns 10
minutos, mas isso varia com a altitude e com a elevação máxima, então
deixar o intervalo fixo ou corta o fim da trajetória ou desenha um arco
comprido em que o satélite está do outro lado do planeta.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from passpredict import Location, Observer, SGP4Propagator

from ..fake_tle import GroundPoint, subpoint

__all__ = [
    "TrackSample",
    "PassMarker",
    "PassTrack",
    "NoPassFoundError",
    "TooManySamplesError",
    "sample_track",
    "build_pass_track",
]

UTC = dt.timezone.utc


class NoPassFoundError(RuntimeError):
    """Nenhuma passagem visível da estação dentro da janela de busca."""


class TooManySamplesError(ValueError):
    """A janela e o passo pedidos gerariam mais amostras que o teto."""


@dataclass(frozen=True, slots=True)
class TrackSample:
    """Um instante da trajetória, visto do chão e do espaço."""

    offset_s: float
    """Segundos desde a época do TLE. Negativo = antes."""

    when: dt.datetime
    lat_deg: float
    lon_deg: float
    alt_km: float
    elevation_deg: float
    azimuth_deg: float
    range_km: float

    @property
    def is_visible(self) -> bool:
        """Acima do horizonte da estação."""
        return self.elevation_deg > 0.0


@dataclass(frozen=True, slots=True)
class PassMarker:
    """Um dos três instantes notáveis da passagem: AOS, TCA ou LOS."""

    when: dt.datetime
    elevation_deg: float
    azimuth_deg: float
    range_km: float


@dataclass(frozen=True, slots=True)
class PassTrack:
    """Trajetória amostrada de uma passagem sobre uma estação."""

    station: GroundPoint
    station_name: str
    epoch: dt.datetime
    samples: tuple[TrackSample, ...]
    aos: PassMarker | None
    tca: PassMarker | None
    los: PassMarker | None

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("uma trajetória precisa de pelo menos uma amostra")

    @property
    def duration_s(self) -> float:
        """Duração da janela amostrada, não da passagem."""
        return self.samples[-1].offset_s - self.samples[0].offset_s

    @property
    def pass_duration_s(self) -> float | None:
        """Tempo entre AOS e LOS, ou None se a passagem não foi localizada."""
        if self.aos is None or self.los is None:
            return None
        return (self.los.when - self.aos.when).total_seconds()

    @property
    def max_elevation_deg(self) -> float:
        """Elevação máxima atingida nas amostras."""
        return max(sample.elevation_deg for sample in self.samples)


def _as_marker(point) -> PassMarker:
    """Converte um `PassPoint` do passpredict no nosso value object.

    O `PassPoint` carrega campos de brilho e visibilidade que só fazem
    sentido para observação óptica; aqui só interessa a geometria.
    """
    return PassMarker(
        when=point.dt.astimezone(UTC),
        elevation_deg=float(point.elevation),
        azimuth_deg=float(point.azimuth),
        range_km=float(point.range),
    )


def sample_track(
    satellite: SGP4Propagator,
    observer: Observer,
    epoch: dt.datetime,
    start: dt.datetime,
    end: dt.datetime,
    step_s: float,
    max_samples: int | None = None,
) -> tuple[TrackSample, ...]:
    """Amostra a trajetória de `start` a `end`, de `step_s` em `step_s`.

    O último instante é sempre incluído, mesmo que a janela não seja um
    múltiplo exato do passo -- sem isso a polyline termina antes do LOS.

    `max_samples` é verificado *antes* de propagar: cada amostra custa
    uma propagação SGP4 mais um razel, então estourar o teto no fim
    significaria ter pago a conta inteira para depois jogá-la fora.
    """
    if step_s <= 0.0:
        raise ValueError(f"step_s precisa ser positivo: {step_s}")
    if end <= start:
        raise ValueError(f"janela vazia: start={start}, end={end}")

    span_s = (end - start).total_seconds()
    steps = int(span_s // step_s)
    offsets = [i * step_s for i in range(steps + 1)]
    if offsets[-1] < span_s:
        offsets.append(span_s)

    if max_samples is not None and len(offsets) > max_samples:
        raise TooManySamplesError(
            f"{len(offsets)} amostras excedem o teto de {max_samples}; "
            f"aumente step_s (atual: {step_s} s) ou reduza a janela"
        )

    samples = []
    for offset in offsets:
        when = start + dt.timedelta(seconds=offset)
        lat, lon, alt = subpoint(satellite, when)
        razel = observer.razel(when)
        samples.append(
            TrackSample(
                offset_s=(when - epoch).total_seconds(),
                when=when,
                lat_deg=lat,
                lon_deg=lon,
                alt_km=alt,
                elevation_deg=float(razel.el),
                azimuth_deg=float(razel.az),
                range_km=float(razel.range),
            )
        )
    return tuple(samples)


def build_pass_track(
    satellite: SGP4Propagator,
    station: GroundPoint,
    epoch: dt.datetime,
    station_name: str = "Ground Station",
    step_s: float = 10.0,
    pad_s: float = 120.0,
    search_back_s: float = 1800.0,
    max_samples: int | None = None,
) -> PassTrack:
    """Localiza a passagem em torno de `epoch` e amostra sua trajetória.

    A busca começa em `epoch - search_back_s` porque os TLEs sintéticos
    deste projeto centram a passagem na época: procurar a partir da
    própria época acharia a passagem seguinte, uma órbita depois.

    `pad_s` estende a janela antes do AOS e depois do LOS, para que a
    ground track apareça no mapa chegando e saindo, em vez de nascer e
    morrer exatamente no horizonte.
    """
    if epoch.tzinfo is None or epoch.utcoffset() is None:
        raise ValueError("epoch precisa ser timezone-aware")

    epoch = epoch.astimezone(UTC)
    observer = Observer(
        Location(station_name, station.lat_deg, station.lon_deg, station.alt_m),
        satellite,
    )

    search_start = epoch - dt.timedelta(seconds=search_back_s)
    overpass = observer.next_pass(
        search_start, limit_date=epoch + dt.timedelta(seconds=search_back_s)
    )
    if overpass is None:
        raise NoPassFoundError(
            f"nenhuma passagem sobre {station_name} "
            f"({station.lat_deg:.4f}, {station.lon_deg:.4f}) numa janela de "
            f"+/-{search_back_s / 60:.0f} min em torno de {epoch:%Y-%m-%d %H:%M:%S} UTC"
        )

    aos, tca, los = (
        _as_marker(overpass.aos),
        _as_marker(overpass.tca),
        _as_marker(overpass.los),
    )
    samples = sample_track(
        satellite,
        observer,
        epoch,
        start=aos.when - dt.timedelta(seconds=pad_s),
        end=los.when + dt.timedelta(seconds=pad_s),
        step_s=step_s,
        max_samples=max_samples,
    )
    return PassTrack(
        station=station,
        station_name=station_name,
        epoch=epoch,
        samples=samples,
        aos=aos,
        tca=tca,
        los=los,
    )
