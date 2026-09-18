"""O formatter é o módulo mais testável: sem física, sem tempo, sem I/O.

O teste que mais paga é o round-trip pelo parser do sgp4: ele valida
posição de coluna sem ninguém contar caractere na mão.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest
from sgp4.api import Satrec

from my_pass_prediction.fake_tle import OrbitalElements, TleIdentity
from my_pass_prediction.fake_tle.formatter import (
    TLE_LINE_LENGTH,
    checksum,
    format_decimal_point_assumed,
    format_epoch,
    format_tle,
)

UTC = dt.timezone.utc


@pytest.fixture
def iss_like_elements() -> OrbitalElements:
    """Elementos próximos aos da ISS, para exercitar valores realistas."""
    return OrbitalElements(
        inc_rad=math.radians(51.6307),
        raan_rad=math.radians(200.0361),
        ecc=0.0004822,
        argp_rad=math.radians(152.4527),
        mean_anomaly_rad=math.radians(207.6718),
        mean_motion_rad_s=2 * math.pi * 15.49160218 / 86400.0,
    )


class TestChecksum:
    def test_bate_com_o_tle_real_da_iss(self, iss_lines):
        """Se esta falhar, as colunas estão deslocadas."""
        for line in iss_lines:
            assert checksum(line) == int(line[TLE_LINE_LENGTH - 1])

    def test_conta_sinal_negativo_como_um(self):
        assert checksum("-" + " " * 67) == 1

    def test_ignora_letras_e_pontos(self):
        assert checksum("U." + " " * 66) == 0


class TestFormatEpoch:
    def test_primeiro_de_janeiro_e_dia_um(self):
        assert format_epoch(dt.datetime(2026, 1, 1, tzinfo=UTC)) == "26001.00000000"

    def test_meio_dia_e_meio_dia_fracionario(self):
        assert format_epoch(dt.datetime(2026, 1, 1, 12, tzinfo=UTC)) == "26001.50000000"

    def test_dia_do_ano_conhecido(self):
        # 2026 não é bissexto: 18/set é o dia 261.
        assert format_epoch(dt.datetime(2026, 9, 18, 12, tzinfo=UTC)) == "26261.50000000"

    def test_converte_para_utc_antes_de_formatar(self):
        brasilia = dt.timezone(dt.timedelta(hours=-3))
        assert format_epoch(dt.datetime(2026, 1, 1, 9, tzinfo=brasilia)) == (
            "26001.50000000"
        )

    def test_recusa_datetime_ingenuo(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            format_epoch(dt.datetime(2026, 1, 1))


class TestDecimalPointAssumed:
    def test_bstar_real_da_iss(self):
        assert format_decimal_point_assumed(0.00010270) == " 10270-3"

    def test_zero(self):
        assert format_decimal_point_assumed(0.0) == " 00000+0"

    def test_negativo(self):
        assert format_decimal_point_assumed(-0.00010270) == "-10270-3"

    def test_sempre_oito_colunas(self):
        for value in (0.0, 1e-8, -2.7e-5, 0.13, -0.9999999):
            assert len(format_decimal_point_assumed(value)) == 8


class TestFormatTle:
    def test_ambas_as_linhas_tem_69_colunas(self, iss_like_elements, epoch):
        for line in format_tle(iss_like_elements, epoch):
            assert len(line) == TLE_LINE_LENGTH

    def test_checksum_das_linhas_geradas_confere(self, iss_like_elements, epoch):
        for line in format_tle(iss_like_elements, epoch):
            assert checksum(line) == int(line[TLE_LINE_LENGTH - 1])

    def test_round_trip_pelo_parser_do_sgp4(self, iss_like_elements, epoch):
        """O teste central: o sgp4 tem que ler de volta o que foi escrito."""
        line1, line2 = format_tle(iss_like_elements, epoch)
        parsed = Satrec.twoline2rv(line1, line2)

        assert parsed.inclo == pytest.approx(iss_like_elements.inc_rad, abs=1e-6)
        assert parsed.nodeo == pytest.approx(iss_like_elements.raan_rad, abs=1e-6)
        assert parsed.ecco == pytest.approx(iss_like_elements.ecc, abs=1e-7)
        assert parsed.argpo == pytest.approx(iss_like_elements.argp_rad, abs=1e-6)
        assert parsed.mo == pytest.approx(iss_like_elements.mean_anomaly_rad, abs=1e-6)

    def test_round_trip_com_inclinacao_de_tres_digitos(self, epoch):
        """Inclinação >= 100 graus é onde o alinhamento de coluna quebra."""
        elements = OrbitalElements(
            inc_rad=math.radians(124.1083),
            raan_rad=math.radians(327.8393),
            ecc=1e-7,
            argp_rad=0.0,
            mean_anomaly_rad=math.radians(195.6347),
            mean_motion_rad_s=1.1068e-3,
        )
        line1, line2 = format_tle(elements, epoch)
        parsed = Satrec.twoline2rv(line1, line2)
        assert math.degrees(parsed.inclo) == pytest.approx(124.1083, abs=1e-4)

    def test_excentricidade_nula_nao_vira_sete_zeros(self, epoch):
        """e = 0 exato deixa alguns caminhos do SGP4 instáveis."""
        elements = OrbitalElements(
            inc_rad=1.0,
            raan_rad=0.0,
            ecc=1e-7,
            argp_rad=0.0,
            mean_anomaly_rad=0.0,
            mean_motion_rad_s=1.1068e-3,
        )
        _, line2 = format_tle(elements, epoch)
        assert line2[26:33] == "0000001"

    def test_identity_vai_para_as_colunas_certas(self, iss_like_elements, epoch):
        identity = TleIdentity(
            satnum=25544, intl_designator="98067A", element_set=999, rev_number=58616
        )
        line1, line2 = format_tle(iss_like_elements, epoch, identity)
        assert line1[2:7] == "25544"
        assert line1[7] == "U"
        assert line1[9:17] == "98067A  "
        assert line1[64:68] == " 999"
        assert line2[2:7] == "25544"
        assert line2[63:68] == "58616"

    def test_campos_de_arrasto_zerados(self, iss_like_elements, epoch):
        line1, _ = format_tle(iss_like_elements, epoch)
        assert line1[33:43] == " .00000000"
        assert line1[44:52] == " 00000+0"
        assert line1[53:61] == " 00000+0"

    def test_angulos_sao_normalizados_para_0_360(self, epoch):
        elements = OrbitalElements(
            inc_rad=1.0,
            raan_rad=-0.1,  # negativo: precisa sair como ~354 graus
            ecc=1e-7,
            argp_rad=0.0,
            mean_anomaly_rad=0.0,
            mean_motion_rad_s=1.1068e-3,
        )
        _, line2 = format_tle(elements, epoch)
        assert float(line2[17:25]) == pytest.approx(360 - math.degrees(0.1), abs=1e-3)
