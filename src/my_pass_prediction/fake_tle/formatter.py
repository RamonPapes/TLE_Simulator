"""Elementos orbitais -> as 69 colunas de um TLE de duas linhas.

Formatador fiel: não corrige nem ajusta física, só escreve o que recebe.
Nenhum import de mecânica orbital aqui -- é o módulo mais testável do
pacote, e o único que precisa ser conferido contra um TLE real.

Layout (índices 0-based, total 69 caracteres por linha):

Linha 1
    0      '1'
    2-6    número de catálogo         7      classificação
    9-16   designador internacional   18-19  ano da época
    20-31  dia do ano fracionário     33-42  ndot/2
    44-51  nddot/6                    53-60  BSTAR
    62     tipo de efeméride          64-67  número do element set
    68     checksum

Linha 2
    0      '2'                        2-6    número de catálogo
    8-15   inclinação                 17-24  RAAN
    26-32  excentricidade             34-41  argumento do perigeu
    43-50  anomalia média             52-62  mean motion (rev/dia)
    63-67  número da revolução        68     checksum
"""

from __future__ import annotations

import datetime as dt
import math

from .models import OrbitalElements, TleIdentity

__all__ = [
    "TLE_LINE_LENGTH",
    "checksum",
    "format_epoch",
    "format_decimal_point_assumed",
    "format_tle",
]

TLE_LINE_LENGTH = 69

# Arrasto desligado: massa e área do satélite não interessam para
# simular distância.
_NO_DRAG_NDOT = " .00000000"
_NO_DRAG_EXPONENTIAL = " 00000+0"


def checksum(line: str) -> int:
    """Soma dos dígitos das 68 primeiras colunas, com '-' valendo 1, mod 10."""
    total = 0
    for char in line[: TLE_LINE_LENGTH - 1]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def format_epoch(when: dt.datetime) -> str:
    """Época no formato do TLE: 'YYDDD.DDDDDDDD' (14 colunas).

    O dia do ano é 1-based: 1 de janeiro 00:00 vira '001.00000000'.
    """
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("`when` precisa ser timezone-aware")
    utc = when.astimezone(dt.timezone.utc)
    year_start = dt.datetime(utc.year, 1, 1, tzinfo=dt.timezone.utc)
    day_of_year = (utc - year_start).total_seconds() / 86400.0 + 1.0
    return f"{utc.year % 100:02d}{day_of_year:012.8f}"


def format_decimal_point_assumed(value: float) -> str:
    """Notação de ponto decimal implícito do TLE: 0.00010270 -> ' 10270-3'."""
    if value == 0.0:
        return _NO_DRAG_EXPONENTIAL
    sign = "-" if value < 0.0 else " "
    magnitude = abs(value)
    exponent = math.floor(math.log10(magnitude)) + 1
    mantissa = round(magnitude / 10.0**exponent * 100_000)
    if mantissa >= 100_000:  # arredondou pra cima e estourou 5 dígitos
        mantissa //= 10
        exponent += 1
    exponent_sign = "-" if exponent < 0 else "+"
    return f"{sign}{mantissa:05d}{exponent_sign}{abs(exponent)}"


def _with_checksum(line68: str) -> str:
    if len(line68) != TLE_LINE_LENGTH - 1:
        raise AssertionError(
            f"linha com {len(line68)} colunas antes do checksum, esperado 68: {line68!r}"
        )
    return f"{line68}{checksum(line68)}"


def format_tle(
    elements: OrbitalElements,
    epoch: dt.datetime,
    identity: TleIdentity | None = None,
) -> tuple[str, str]:
    """As duas linhas do TLE, checksum incluído."""
    identity = identity or TleIdentity()

    # Excentricidade: 7 dígitos com ponto decimal implícito ("0006317").
    eccentricity = f"{elements.ecc:.7f}"[2:]
    inclination = f"{math.degrees(elements.inc_rad) % 360.0:8.4f}"
    raan = f"{math.degrees(elements.raan_rad) % 360.0:8.4f}"
    arg_perigee = f"{math.degrees(elements.argp_rad) % 360.0:8.4f}"
    mean_anomaly = f"{math.degrees(elements.mean_anomaly_rad) % 360.0:8.4f}"
    mean_motion = f"{elements.mean_motion_rev_per_day:11.8f}"

    line1 = _with_checksum(
        f"1 {identity.satnum:05d}{identity.classification} "
        f"{identity.intl_designator:<8} "
        f"{format_epoch(epoch)} "
        f"{_NO_DRAG_NDOT} {_NO_DRAG_EXPONENTIAL} {_NO_DRAG_EXPONENTIAL} 0 "
        f"{identity.element_set:4d}"
    )
    line2 = _with_checksum(
        f"2 {identity.satnum:05d} "
        f"{inclination} {raan} {eccentricity} "
        f"{arg_perigee} {mean_anomaly} "
        f"{mean_motion}{identity.rev_number:5d}"
    )
    return line1, line2
