"""Correções de bugs do passpredict 0.5.1.

Mesmo espírito do `CelestrakOrgTLESource` em demo_celestrak.py: contornar
defeito de biblioteca num lugar só, bem sinalizado.
"""

from __future__ import annotations

from passpredict import TLE

__all__ = ["PatchedTLE"]


class PatchedTLE(TLE):
    """TLE com a coluna de inclinação lida corretamente.

    O passpredict 0.5.1 faz `float(self.tle2[9:17])`, mas a inclinação
    ocupa as colunas 9-16 do padrão (índices 8-15). O deslocamento de uma
    coluna é inofensivo abaixo de 100 graus, porque o campo vem com espaço
    à esquerda e o `float` engole o espaço à direita. A partir de 100
    graus ele descarta o dígito de centena em silêncio: uma órbita de
    124.1 graus é propagada como 24.1, sem erro nem aviso.

    Só a inclinação está errada; RAAN, excentricidade, argumento do
    perigeu, anomalia média e mean motion usam colunas corretas.
    """

    @property
    def inc(self) -> float:
        return float(self.tle2[8:16])
