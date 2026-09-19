"""Fronteira da web: parâmetros do formulário -> dicionário JSON.

Esta camada faz duas coisas que nenhuma outra faz: valida entrada que
veio de fora (e portanto não é confiável) e decide o formato do fio. O
núcleo continua ignorando que existe um navegador.

Não importa `http`: `compute_pass` é uma função pura de dict para dict,
o que a torna testável sem subir servidor nenhum.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Mapping

from passpredict import SGP4Propagator

from ..fake_tle import (
    ConvergenceError,
    DegenerateGeometryError,
    GroundPoint,
    PassSolution,
    TleIdentity,
    generate_fake_tle,
    point_at_bearing,
)
from .track import (
    NoPassFoundError,
    PassMarker,
    PassTrack,
    TooManySamplesError,
    build_pass_track,
)

__all__ = [
    "InvalidRequestError",
    "PassRequest",
    "compute_pass",
    "DEFAULTS",
    "MAX_SAMPLES",
]

UTC = dt.timezone.utc

# Teto de amostras por requisição. Cada amostra custa uma propagação
# SGP4 mais um razel; sem limite, uma janela larga com passo fino
# prenderia a thread do servidor por muito tempo.
MAX_SAMPLES = 5000

DEFAULTS: dict[str, Any] = {
    "lat": -12.9777,
    "lon": -38.5016,
    "alt_m": 0.0,
    "name": "Salvador",
    "bearing_deg": 200.0,
    "arc_deg": 20.0,
    "altitude_km": 500.0,
    "step_s": 10.0,
    "pad_s": 120.0,
}


class InvalidRequestError(ValueError):
    """Entrada do usuário fora do domínio aceito."""


def _as_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    """Lê um número do payload, aceitando string (é o que o form manda)."""
    raw = payload.get(key, default)
    if raw is None or raw == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InvalidRequestError(f"{key!r} precisa ser um número: {raw!r}") from None
    if not math.isfinite(value):
        raise InvalidRequestError(f"{key!r} precisa ser finito: {raw!r}")
    return value


def _in_range(key: str, value: float, low: float, high: float) -> float:
    if not low <= value <= high:
        raise InvalidRequestError(f"{key} fora de [{low}, {high}]: {value}")
    return value


def _parse_epoch(raw: Any) -> dt.datetime:
    """Aceita ISO 8601; sem timezone, assume UTC.

    O input `datetime-local` do HTML manda sempre um instante ingênuo
    ("2026-09-18T12:00"), e `fromisoformat` no Python 3.10 ainda não
    entende o sufixo "Z" -- daí a troca explícita.
    """
    if raw is None or raw == "":
        return dt.datetime.now(UTC).replace(microsecond=0)
    if not isinstance(raw, str):
        raise InvalidRequestError(f"'epoch' precisa ser uma string ISO 8601: {raw!r}")
    try:
        parsed = dt.datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        raise InvalidRequestError(
            f"'epoch' não é uma data ISO 8601 válida: {raw!r}"
        ) from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PassRequest:
    """Parâmetros validados de uma simulação de passagem."""

    station: GroundPoint
    station_name: str
    epoch: dt.datetime
    bearing_deg: float
    arc_deg: float
    altitude_km: float
    step_s: float
    pad_s: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PassRequest:
        """Constrói a requisição a partir do JSON cru do navegador."""
        if not isinstance(payload, Mapping):
            raise InvalidRequestError("payload precisa ser um objeto JSON")

        lat = _in_range("lat", _as_float(payload, "lat", DEFAULTS["lat"]), -90.0, 90.0)
        lon = _in_range(
            "lon", _as_float(payload, "lon", DEFAULTS["lon"]), -180.0, 180.0
        )
        alt_m = _in_range(
            "alt_m", _as_float(payload, "alt_m", DEFAULTS["alt_m"]), -500.0, 9000.0
        )
        # A órbita precisa ficar acima da atmosfera densa e abaixo do
        # regime em que uma "passagem" deixa de ser um evento curto.
        altitude_km = _in_range(
            "altitude_km",
            _as_float(payload, "altitude_km", DEFAULTS["altitude_km"]),
            150.0,
            2000.0,
        )
        step_s = _in_range(
            "step_s", _as_float(payload, "step_s", DEFAULTS["step_s"]), 1.0, 300.0
        )
        pad_s = _in_range(
            "pad_s", _as_float(payload, "pad_s", DEFAULTS["pad_s"]), 0.0, 1800.0
        )
        # `arc_deg` é a separação angular entre origem e ponto de mira.
        # Perto de 0 ou de 180 graus os dois pontos não definem um plano
        # único, e o solver levantaria DegenerateGeometryError.
        arc_deg = _in_range(
            "arc_deg", _as_float(payload, "arc_deg", DEFAULTS["arc_deg"]), 1.0, 179.0
        )
        bearing_deg = _as_float(payload, "bearing_deg", DEFAULTS["bearing_deg"]) % 360.0

        name = str(payload.get("name") or DEFAULTS["name"]).strip()[:64]

        return cls(
            station=GroundPoint(lat, lon, alt_m),
            station_name=name or DEFAULTS["name"],
            epoch=_parse_epoch(payload.get("epoch")),
            bearing_deg=bearing_deg,
            arc_deg=arc_deg,
            altitude_km=altitude_km,
            step_s=step_s,
            pad_s=pad_s,
        )

    @property
    def heading(self) -> GroundPoint:
        """Ponto de mira que fixa a direção da ground track."""
        lat, lon = point_at_bearing(
            self.station.lat_deg, self.station.lon_deg, self.bearing_deg, self.arc_deg
        )
        return GroundPoint(lat, lon)


def _marker_dict(marker: PassMarker | None) -> dict[str, Any] | None:
    if marker is None:
        return None
    return {
        "t": marker.when.isoformat(),
        "el": round(marker.elevation_deg, 3),
        "az": round(marker.azimuth_deg, 3),
        "range_km": round(marker.range_km, 3),
    }


def _track_dict(track: PassTrack) -> dict[str, Any]:
    """Serializa a trajetória.

    As chaves são curtas e os floats arredondados de propósito: uma
    passagem amostrada de 1 em 1 s passa de 800 pontos, e cada casa
    decimal a mais é peso no fio sem ganho visível no mapa (5 casas em
    grau valem cerca de 1 m).
    """
    return {
        "samples": [
            {
                "t": round(sample.offset_s, 3),
                "lat": round(sample.lat_deg, 5),
                "lon": round(sample.lon_deg, 5),
                "alt_km": round(sample.alt_km, 3),
                "el": round(sample.elevation_deg, 3),
                "az": round(sample.azimuth_deg, 3),
                "range_km": round(sample.range_km, 3),
            }
            for sample in track.samples
        ],
        "aos": _marker_dict(track.aos),
        "tca": _marker_dict(track.tca),
        "los": _marker_dict(track.los),
        "pass_duration_s": (
            None
            if track.pass_duration_s is None
            else round(track.pass_duration_s, 3)
        ),
        "max_elevation_deg": round(track.max_elevation_deg, 3),
    }


def _orbit_dict(solution: PassSolution, altitude_km: float) -> dict[str, Any]:
    elements = solution.elements
    return {
        "inclination_deg": round(math.degrees(elements.inc_rad), 4),
        "raan_deg": round(math.degrees(elements.raan_rad), 4),
        "period_min": round(elements.period_s / 60.0, 4),
        "mean_motion_rev_day": round(elements.mean_motion_rev_per_day, 6),
        "altitude_km": round(altitude_km, 3),
        "iterations": solution.iterations,
    }


def compute_pass(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Gera o TLE sintético, amostra a passagem e devolve o dict JSON.

    Erros do núcleo (geometria degenerada, não-convergência, nenhuma
    passagem encontrada) viram `InvalidRequestError`: do ponto de vista
    do navegador são todos "esses parâmetros não produzem uma passagem",
    e não falhas do servidor.
    """
    request = PassRequest.from_payload(payload)

    try:
        tle, solution = generate_fake_tle(
            request.station,
            request.heading,
            request.epoch,
            altitude_km=request.altitude_km,
            identity=TleIdentity(name=f"FAKE-{request.station_name.upper()[:8]}"),
        )
    except (DegenerateGeometryError, ConvergenceError) as exc:
        raise InvalidRequestError(f"não foi possível montar a órbita: {exc}") from exc

    satellite = SGP4Propagator.from_tle(tle)
    try:
        track = build_pass_track(
            satellite,
            request.station,
            request.epoch,
            station_name=request.station_name,
            step_s=request.step_s,
            pad_s=request.pad_s,
            max_samples=MAX_SAMPLES,
        )
    except (NoPassFoundError, TooManySamplesError) as exc:
        raise InvalidRequestError(str(exc)) from exc

    return {
        "tle": {"name": tle.name, "line1": tle.tle1, "line2": tle.tle2},
        "orbit": _orbit_dict(solution, request.altitude_km),
        "station": {
            "name": request.station_name,
            "lat": request.station.lat_deg,
            "lon": request.station.lon_deg,
            "alt_m": request.station.alt_m,
        },
        "heading": {"lat": request.heading.lat_deg, "lon": request.heading.lon_deg},
        "epoch": request.epoch.isoformat(),
        "request": {
            "bearing_deg": request.bearing_deg,
            "arc_deg": request.arc_deg,
            "altitude_km": request.altitude_km,
            "step_s": request.step_s,
            "pad_s": request.pad_s,
        },
        "track": _track_dict(track),
    }
