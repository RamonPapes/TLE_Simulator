"""Passagem sintética sobre Salvador, BA, rastreável pelo SGP4.

Receita: usar Salvador como ORIGEM (satélite no zênite na época) e um
segundo ponto só para escolher a direção da passagem. O segundo ponto não
precisa ser uma cidade -- `point_at_bearing` gera um ponto a uma distância
e azimute dados, o que deixa a geometria da passagem como parâmetro.

    uv run python examples/pass_over_salvador.py
"""

from __future__ import annotations

import datetime as dt
import math

from passpredict import Location, Observer, SGP4Propagator

from my_pass_prediction.fake_tle import GroundPoint, generate_fake_tle, subpoint

SALVADOR = GroundPoint(-12.9777, -38.5016)
UTC = dt.timezone.utc


def point_at_bearing(
    origin: GroundPoint, bearing_deg: float, arc_deg: float
) -> GroundPoint:
    """Ponto a `arc_deg` de distância angular, no azimute `bearing_deg`.

    Navegação por grande círculo. Serve para escolher por onde a órbita
    passa sem precisar caçar uma cidade conveniente no mapa: o azimute
    vira a direção da ground track sobre a origem.
    """
    lat1, lon1 = origin.lat_rad, origin.lon_rad
    bearing, arc = math.radians(bearing_deg), math.radians(arc_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(arc)
        + math.cos(lat1) * math.sin(arc) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(arc) * math.cos(lat1),
        math.cos(arc) - math.sin(lat1) * math.sin(lat2),
    )
    return GroundPoint(math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0)


def main() -> None:
    # A época é o instante do zênite. Escolha livre.
    epoch = dt.datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

    # Azimute 200 graus = ground track rumo sul-sudoeste (passagem
    # descendente). Troque para 20 graus e ela vira ascendente.
    heading = point_at_bearing(SALVADOR, bearing_deg=200.0, arc_deg=20.0)

    tle, solution = generate_fake_tle(SALVADOR, heading, epoch, altitude_km=500.0)

    print("TLE gerado")
    print(tle.tle1)
    print(tle.tle2)
    print(
        f"\ninclinação {math.degrees(solution.elements.inc_rad):7.3f}°"
        f"   período {solution.elements.period_s / 60:.2f} min"
        f"   zênite em {epoch:%Y-%m-%d %H:%M:%S} UTC"
    )

    satellite = SGP4Propagator.from_tle(tle)
    observer = Observer(
        Location("Salvador", SALVADOR.lat_deg, SALVADOR.lon_deg, 0.0), satellite
    )

    # A passagem é CENTRADA na época, então o início dela é ANTES.
    # Procurar a partir da própria época encontraria a passagem seguinte.
    search_start = epoch - dt.timedelta(minutes=20)
    overpass = observer.get_next_pass(search_start)

    print(
        f"\npassagem: AOS {overpass.aos.dt:%H:%M:%S}"
        f"  TCA {overpass.tca.dt:%H:%M:%S} ({overpass.tca.elevation:.2f}°)"
        f"  LOS {overpass.los.dt:%H:%M:%S}"
        f"  duração {overpass.los.dt - overpass.aos.dt}"
    )

    print("\n  t (s)   elev (°)   alcance (km)   subponto")
    for offset in range(-300, 301, 60):
        when = epoch + dt.timedelta(seconds=offset)
        razel = observer.razel(when)
        lat, lon, _ = subpoint(satellite, when)
        print(
            f"  {offset:+5d}   {razel.el:7.2f}   {razel.range:10.1f}"
            f"     {lat:7.3f}, {lon:8.3f}"
        )


if __name__ == "__main__":
    main()
