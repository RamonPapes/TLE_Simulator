# Python 3.10 não é preferência, é requisito: o `passpredict` publica
# wheels só para cp38/cp39/cp310 (veja [tool.cibuildwheel] no pyproject
# dele). Em 3.11+ o pip cai no sdist e tenta compilar extensão C com
# Cython, numpy e scipy -- que é justamente o que esta imagem evita.
FROM python:3.10-slim AS builder

# uv vem da imagem oficial dele, sem precisar de pip install.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1

WORKDIR /app

# Dependências antes do código: enquanto o uv.lock não mudar, o Docker
# reaproveita esta camada e o rebuild não baixa numpy de novo.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --no-dev


# Segundo estágio só para deixar o cache do uv (270 MB) para trás, junto
# com o próprio uv. O venv basta: `my-pass-web` mora em .venv/bin.
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Sem navegador: não há um dentro do container.
CMD ["my-pass-web", "--no-browser"]
