"""Servidor HTTP da interface, em cima da stdlib.

Duas rotas: arquivos de `static/` e `POST /api/pass`. Nenhuma
dependência nova -- `http.server` basta para uma ferramenta local, e o
custo de subir FastAPI aqui seria maior que o ganho.

    uv run my-pass-web
    uv run my-pass-web --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import pathlib
import socketserver
import sys
import threading
import webbrowser
from typing import Any

from .api import InvalidRequestError, compute_pass

__all__ = ["PassRequestHandler", "serve", "main"]

STATIC_DIR = pathlib.Path(__file__).parent / "static"
INDEX = "index.html"

# Teto do corpo de uma requisição. O payload legítimo tem algumas
# centenas de bytes; ler sem limite deixaria um POST gigante consumir
# memória do processo à vontade.
MAX_BODY_BYTES = 64 * 1024


class PassRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serve os arquivos estáticos e a rota de cálculo."""

    server_version = "PassSimulator/0.1"
    protocol_version = "HTTP/1.1"

    quiet: bool = False

    # ---------------------------------------------------------------- #
    # Escrita de respostas
    # ---------------------------------------------------------------- #

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A página é gerada a cada requisição e o servidor é local:
        # cache aqui só atrapalha quem está editando o CSS ou o JS.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    # ---------------------------------------------------------------- #
    # Rotas
    # ---------------------------------------------------------------- #

    def _resolve_static(self, path: str) -> pathlib.Path | None:
        """Mapeia o caminho da URL para um arquivo dentro de `static/`.

        Devolve None para qualquer coisa que escape do diretório. O
        `SimpleHTTPRequestHandler` já faria essa normalização, mas ele
        serve a partir do diretório de trabalho do processo -- que numa
        ferramenta de linha de comando é o repositório inteiro.
        """
        relative = path.lstrip("/") or INDEX
        candidate = (STATIC_DIR / relative).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def do_GET(self) -> None:  # noqa: N802 (nome exigido pela stdlib)
        route = self.path.split("?", 1)[0]
        if route in ("/api/pass", "/api/pass/"):
            self._send_error_json(405, "use POST em /api/pass")
            return

        target = self._resolve_static(route)
        if target is None:
            self._send(404, b"404 not found\n", "text/plain; charset=utf-8")
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in ("/api/pass", "/api/pass/"):
            self._send_error_json(404, f"rota desconhecida: {self.path}")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_error_json(400, "Content-Length inválido")
            return
        if length > MAX_BODY_BYTES:
            self._send_error_json(413, f"corpo maior que {MAX_BODY_BYTES} bytes")
            return

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error_json(400, f"JSON inválido: {exc}")
            return

        try:
            self._send_json(200, compute_pass(payload))
        except InvalidRequestError as exc:
            self._send_error_json(400, str(exc))
        except Exception as exc:  # pragma: no cover - rede de segurança
            # Um erro inesperado no núcleo não pode derrubar o servidor:
            # o usuário só mudaria um parâmetro e tentaria de novo.
            self.log_error("erro ao calcular a passagem: %r", exc)
            self._send_error_json(500, f"erro interno: {exc}")

    # ---------------------------------------------------------------- #

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.quiet:
            super().log_message(fmt, *args)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    """Uma thread por conexão, com a porta reutilizável ao reiniciar."""

    daemon_threads = True
    allow_reuse_address = True


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    quiet: bool = False,
) -> None:
    """Sobe o servidor e bloqueia até Ctrl-C.

    `port=0` deixa o sistema escolher uma porta livre, o que evita o
    erro de porta ocupada quando outra instância já está no ar.
    """
    PassRequestHandler.quiet = quiet

    with _ThreadingServer((host, port), PassRequestHandler) as httpd:
        actual_port = httpd.server_address[1]
        url = f"http://{host}:{actual_port}/"
        print(f"simulador de passagens em {url}  (Ctrl-C para sair)")

        if open_browser:
            # Numa thread: `webbrowser.open` pode bloquear enquanto o
            # navegador sobe, e aí a primeira requisição chegaria antes
            # do `serve_forever`.
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nencerrando")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="my-pass-web",
        description="Interface web do simulador de passagens com TLE sintético.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="padrão: 127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=8000, help="padrão: 8000; use 0 para escolher livre"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="não abrir o navegador"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="sem log de acesso")
    args = parser.parse_args(argv)

    try:
        serve(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            quiet=args.quiet,
        )
    except OSError as exc:
        print(f"não foi possível abrir {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
