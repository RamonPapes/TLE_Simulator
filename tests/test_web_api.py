"""Fronteira da web: validação de entrada e formato do fio.

`compute_pass` é dict -> dict, então dá para testar a API inteira sem
subir servidor. Os testes de rota, no fim, sobem um de verdade numa
porta efêmera.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from my_pass_prediction.web.api import (
    DEFAULTS,
    InvalidRequestError,
    PassRequest,
    compute_pass,
)

UTC = dt.timezone.utc

BASE_PAYLOAD = {
    "lat": -12.9777,
    "lon": -38.5016,
    "name": "Salvador",
    "epoch": "2026-09-18T12:00:00",
    "bearing_deg": 200,
    "arc_deg": 20,
    "altitude_km": 500,
    "step_s": 10,
    "pad_s": 120,
}


class TestPassRequest:
    def test_payload_vazio_usa_os_padroes(self):
        request = PassRequest.from_payload({})
        assert request.station.lat_deg == DEFAULTS["lat"]
        assert request.altitude_km == DEFAULTS["altitude_km"]
        assert request.epoch.tzinfo is not None

    def test_aceita_numeros_como_string(self):
        """É o que um `<input>` manda, mesmo com type=number."""
        request = PassRequest.from_payload({"lat": "-23.55", "altitude_km": "700"})
        assert request.station.lat_deg == pytest.approx(-23.55)
        assert request.altitude_km == pytest.approx(700.0)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("lat", 91),
            ("lat", -91),
            ("lon", 181),
            ("altitude_km", 100),
            ("altitude_km", 2001),
            ("step_s", 0.5),
            ("step_s", 301),
            ("pad_s", -1),
            ("arc_deg", 0.5),
            ("arc_deg", 179.5),
        ],
    )
    def test_fora_de_faixa_levanta(self, field, value):
        with pytest.raises(InvalidRequestError, match=field):
            PassRequest.from_payload({field: value})

    # "12,5" é o caso realista: vírgula decimal, que `float()` recusa.
    @pytest.mark.parametrize("value", ["abc", "12,5", [1, 2], {}])
    def test_nao_numero_levanta(self, value):
        with pytest.raises(InvalidRequestError, match="precisa ser um número"):
            PassRequest.from_payload({"lat": value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nao_finito_levanta(self, value):
        """NaN passa por `float()` mas envenenaria toda a geometria adiante."""
        with pytest.raises(InvalidRequestError, match="precisa ser finito"):
            PassRequest.from_payload({"lat": value})

    @pytest.mark.parametrize("value", [None, ""])
    def test_campo_ausente_ou_vazio_cai_no_padrao(self, value):
        assert PassRequest.from_payload({"lat": value}).station.lat_deg == DEFAULTS["lat"]

    def test_epoca_ingenua_vira_utc(self):
        request = PassRequest.from_payload({"epoch": "2026-09-18T12:00:00"})
        assert request.epoch == dt.datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

    def test_sufixo_z_e_aceito(self):
        """`fromisoformat` no Python 3.10 ainda não entende o 'Z'."""
        request = PassRequest.from_payload({"epoch": "2026-09-18T12:00:00Z"})
        assert request.epoch == dt.datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

    def test_offset_e_convertido_para_utc(self):
        request = PassRequest.from_payload({"epoch": "2026-09-18T09:00:00-03:00"})
        assert request.epoch == dt.datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

    def test_data_invalida_levanta(self):
        with pytest.raises(InvalidRequestError, match="ISO 8601"):
            PassRequest.from_payload({"epoch": "ontem de tarde"})

    def test_azimute_e_reduzido_ao_circulo(self):
        assert PassRequest.from_payload({"bearing_deg": 380}).bearing_deg == 20.0
        assert PassRequest.from_payload({"bearing_deg": -10}).bearing_deg == 350.0

    def test_nome_longo_e_truncado(self):
        request = PassRequest.from_payload({"name": "x" * 200})
        assert len(request.station_name) == 64

    def test_nome_vazio_cai_no_padrao(self):
        assert PassRequest.from_payload({"name": "   "}).station_name == DEFAULTS["name"]

    def test_heading_fica_no_azimute_pedido(self):
        """Azimute 180 graus a partir do equador desce pelo mesmo meridiano."""
        request = PassRequest.from_payload(
            {"lat": 0, "lon": 0, "bearing_deg": 180, "arc_deg": 20}
        )
        assert request.heading.lat_deg == pytest.approx(-20.0)
        assert request.heading.lon_deg == pytest.approx(0.0, abs=1e-9)


@pytest.mark.integration
class TestComputePass:
    @staticmethod
    @pytest.fixture(scope="class")
    def result():
        return compute_pass(BASE_PAYLOAD)

    def test_devolve_as_duas_linhas_do_tle(self, result):
        assert result["tle"]["line1"].startswith("1 ")
        assert result["tle"]["line2"].startswith("2 ")
        assert len(result["tle"]["line1"]) == 69
        assert len(result["tle"]["line2"]) == 69

    def test_o_nome_do_satelite_vem_da_estacao(self, result):
        assert result["tle"]["name"] == "FAKE-SALVADOR"

    def test_passagem_pelo_zenite(self, result):
        assert result["track"]["tca"]["el"] > 85.0

    def test_amostras_no_passo_pedido(self, result):
        samples = result["track"]["samples"]
        assert len(samples) > 10
        assert samples[1]["t"] - samples[0]["t"] == pytest.approx(10.0)

    def test_a_trilha_comeca_e_termina_abaixo_do_horizonte(self, result):
        """Efeito da folga: a ground track entra e sai do campo de visão."""
        samples = result["track"]["samples"]
        assert samples[0]["el"] < 0.0
        assert samples[-1]["el"] < 0.0
        assert any(s["el"] > 0.0 for s in samples)

    def test_resposta_e_serializavel(self, result):
        """O servidor vai fazer json.dumps nisso; nada de NumPy escapando."""
        assert json.loads(json.dumps(result)) == result

    def test_passo_fino_demais_excede_o_teto(self):
        with pytest.raises(InvalidRequestError, match="excedem o teto"):
            compute_pass({**BASE_PAYLOAD, "step_s": 1, "pad_s": 1800, "altitude_km": 2000})


class TestConfiguracaoPorAmbiente:
    """A hospedagem escolhe a porta e a anuncia em $PORT.

    Um container que ignore isso sobe, passa no build e nunca recebe
    tráfego -- falha silenciosa, que é a pior de depurar.
    """

    @staticmethod
    def _reload(monkeypatch, **env):
        import importlib

        from my_pass_prediction.web import server

        for key in ("HOST", "PORT"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(server)

    def test_sem_ambiente_fica_no_localhost(self, monkeypatch):
        server = self._reload(monkeypatch)
        assert server.DEFAULT_HOST == "127.0.0.1"
        assert server.DEFAULT_PORT == 8000

    def test_porta_e_host_vem_do_ambiente(self, monkeypatch):
        server = self._reload(monkeypatch, PORT="10000", HOST="0.0.0.0")
        assert server.DEFAULT_PORT == 10000
        assert server.DEFAULT_HOST == "0.0.0.0"

    def test_a_linha_de_comando_vence_o_ambiente(self, monkeypatch):
        parser = self._reload(monkeypatch, PORT="10000", HOST="0.0.0.0").build_parser()
        assert parser.parse_args([]).port == 10000
        assert parser.parse_args(["--port", "9999"]).port == 9999
        assert parser.parse_args(["--host", "localhost"]).host == "localhost"


@pytest.mark.integration
class TestServerRoutes:
    @staticmethod
    @pytest.fixture(scope="class")
    def base_url():
        from my_pass_prediction.web.server import PassRequestHandler, _ThreadingServer

        PassRequestHandler.quiet = True
        with _ThreadingServer(("127.0.0.1", 0), PassRequestHandler) as httpd:
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{httpd.server_address[1]}"
            finally:
                httpd.shutdown()
                thread.join(timeout=5)

    def _post(self, url: str, payload: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_raiz_serve_o_index(self, base_url):
        with urllib.request.urlopen(f"{base_url}/", timeout=10) as response:
            body = response.read().decode()
        assert response.status == 200
        assert "<title>" in body and 'src="app.js"' in body

    @pytest.mark.parametrize(
        "asset", ["app.js", "style.css", "satellite.svg", "ground-station.svg"]
    )
    def test_serve_os_estaticos(self, base_url, asset):
        with urllib.request.urlopen(f"{base_url}/{asset}", timeout=10) as response:
            assert response.status == 200
            assert len(response.read()) > 0

    def test_todo_icone_referenciado_pelo_js_existe(self, base_url):
        """Renomear um SVG sem atualizar o JS quebraria o mapa em silêncio.

        O Leaflet não avisa: o marcador simplesmente não aparece.
        """
        with urllib.request.urlopen(f"{base_url}/app.js", timeout=10) as response:
            app_js = response.read().decode()

        referenced = re.findall(r'iconUrl:\s*"([^"]+)"', app_js)
        assert referenced, "nenhum iconUrl encontrado em app.js"

        for icon in referenced:
            with urllib.request.urlopen(f"{base_url}/{icon}", timeout=10) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "image/svg+xml"

    @pytest.mark.parametrize(
        "path", ["/../api.py", "/../../pyproject.toml", "/nao-existe.txt"]
    )
    def test_nao_serve_nada_fora_de_static(self, base_url, path):
        """O diretório servido é `static/`, não o diretório de trabalho."""
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base_url}{path}", timeout=10)
        assert exc.value.code == 404

    def test_get_na_api_responde_405(self, base_url):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base_url}/api/pass", timeout=10)
        assert exc.value.code == 405

    def test_post_valido_devolve_a_passagem(self, base_url):
        status, body = self._post(f"{base_url}/api/pass", BASE_PAYLOAD)
        assert status == 200
        assert body["track"]["tca"]["el"] > 85.0

    def test_post_invalido_devolve_400_com_mensagem(self, base_url):
        status, body = self._post(f"{base_url}/api/pass", {"lat": 999})
        assert status == 400
        assert "lat" in body["error"]

    def test_json_quebrado_devolve_400(self, base_url):
        request = urllib.request.Request(
            f"{base_url}/api/pass", data=b"{nao e json", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=10)
        assert exc.value.code == 400

    def test_rota_desconhecida_devolve_404(self, base_url):
        status, body = self._post(f"{base_url}/api/inexistente", {})
        assert status == 404
        assert "error" in body
