"""Compatibilidade: a implementação mudou para o pacote `fake_tle`.

Mantido só para não quebrar imports antigos. Prefira importar direto de
`my_pass_prediction.fake_tle`.
"""

from __future__ import annotations

from .fake_tle import generate_fake_tle

__all__ = ["generate_fake_tle"]
