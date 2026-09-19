"""Interface web: mapa animado da passagem do satélite.

Três camadas, na ordem em que os dados fluem:

- `track`: órbita -> série temporal (lat, lon, elevação, alcance)
- `api`:   parâmetros do usuário -> TLE sintético -> track -> dict JSON
- `server`: HTTP em cima de `http.server`, servindo `static/`

    uv run my-pass-web
"""

from __future__ import annotations

from .api import InvalidRequestError, PassRequest, compute_pass
from .server import main, serve
from .track import (
    NoPassFoundError,
    TooManySamplesError,
    PassMarker,
    PassTrack,
    TrackSample,
    build_pass_track,
    sample_track,
)

__all__ = [
    "compute_pass",
    "PassRequest",
    "InvalidRequestError",
    "serve",
    "main",
    "TrackSample",
    "PassMarker",
    "PassTrack",
    "NoPassFoundError",
    "TooManySamplesError",
    "sample_track",
    "build_pass_track",
]
