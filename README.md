# my-pass-prediction

Gera TLEs sintéticos que produzem uma passagem sobre um ponto escolhido, e
mostra essa passagem num mapa animado.

A ideia: em vez de procurar um satélite real que por acaso passe onde você
precisa, você diz onde e quando quer a passagem e o gerador resolve a órbita
que a satisfaz. O TLE resultante é um TLE de verdade -- propaga no SGP4,
carrega no `passpredict`, no GPredict ou em qualquer coisa que leia o formato.

## Interface web

```bash
uv run my-pass-web
```

Sobe um servidor local em <http://127.0.0.1:8000> e abre o navegador. O mapa
mostra a ground track da passagem, com o trecho visível da estação em
destaque, e a barra de baixo reproduz o sobrevoo.

| opção | efeito |
|---|---|
| `--port 8080` | outra porta (`--port 0` escolhe uma livre) |
| `--host 0.0.0.0` | aceita conexões de fora da máquina |
| `--no-browser` | não abre o navegador |
| `-q` | sem log de acesso |

Parâmetros na própria página: posição da estação (ou clique no mapa), época
do zênite, altitude, azimute da ground track, passo e folga da amostragem.

## Linha de comando

```bash
uv run python examples/pass_over_salvador.py
```

Imprime o TLE, os elementos, AOS/TCA/LOS e uma tabela de elevação e alcance.

## Como usar como biblioteca

```python
import datetime as dt
from my_pass_prediction.fake_tle import GroundPoint, generate_fake_tle

salvador = GroundPoint(-12.9777, -38.5016)
sao_paulo = GroundPoint(-23.5505, -46.6333)
epoch = dt.datetime(2026, 9, 18, 12, 0, tzinfo=dt.timezone.utc)

tle, solution = generate_fake_tle(salvador, sao_paulo, epoch, altitude_km=500.0)
print(tle.tle1, tle.tle2, sep="\n")
```

O satélite fica no zênite de `salvador` na época e passa sobre `sao_paulo`
depois. Para outros critérios -- "elevação máxima de 40° sobre B", por
exemplo -- passe um `PlaneTarget` em `plane_target`; `MaxElevationTarget` já
vem pronto.

## Organização

```
src/my_pass_prediction/
  fake_tle/          núcleo: resolve a órbita e escreve o TLE
    models.py        value objects; só stdlib
    geometry.py      vetores, planos e ângulos; sem tempo e sem I/O
    targets.py       critérios de escolha do plano orbital (ponto de extensão)
    solver.py        itera até a órbita bater no alvo
    frames.py        modelos de Terra e tempo sideral
    formatter.py     colunas e checksum do formato TLE
    generator.py     composition root; único módulo que importa passpredict
  web/               interface
    track.py         órbita -> série temporal (lat, lon, elevação, alcance)
    api.py           entrada do usuário -> TLE -> track -> dict JSON
    server.py        http.server; duas rotas
    static/          index.html, app.js, style.css
```

A interface web não acrescenta nenhuma dependência: `http.server` da stdlib
serve os arquivos, e o Leaflet vem de CDN.

## Deploy

O `Dockerfile` fixa **Python 3.10**, e isso não é preferência: o
`passpredict` publica wheels só para cp38/cp39/cp310 (veja
`[tool.cibuildwheel]` no `pyproject.toml` dele). Em 3.11+ o `pip` cai no
sdist e tenta compilar extensão C com Cython, numpy e scipy.

Isso descarta plataformas que impõem a versão do Python -- a Vercel, por
exemplo, só oferece 3.12 e não aceita Docker. Serve qualquer lugar que
rode container: Render, Railway, Fly.io, Hugging Face Spaces.

```bash
docker build -t tle-simulator .
docker run --rm -p 8000:8000 tle-simulator
```

Em produção a porta vem de `$PORT` e o host de `$HOST`, que é o que as
plataformas injetam. O `render.yaml` na raiz configura a Render sozinho.

Não há autenticação: qualquer um com a URL calcula passagens. O limite
de amostras por requisição (`MAX_SAMPLES`, em `web/api.py`) é o que
impede uma requisição de prender o processo.

## Testes

```bash
uv run pytest                      # tudo
uv run pytest -m "not integration" # sem propagação SGP4
```
