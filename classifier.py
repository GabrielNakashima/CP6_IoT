"""Classificador de repeticao (BOA x RUIM/FADIGA) - Smart Gym / CP6.

Carrega o modelo treinado por `train_model.py` e expoe uma interface simples:

    classificador = ClassificadorRepeticoes()
    resultado = classificador.classificar(features)
    resultado.classificacao   # "BOA" ou "RUIM"
    resultado.probabilidade   # 0..1 (confianca de ser BOA)

Se o modelo ainda nao existir (ou estiver corrompido), o sistema entra em
"modo heuristico": uma regra de bolso baseada nas mesmas features, para que a
demonstracao nunca fique sem resposta. O modo usado aparece no dashboard e no
banco de dados.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from features import FEATURE_NAMES

MODELO_PADRAO = "modelo_repeticao.joblib"
LIMIAR_BOA = 0.5

NOMES_MODELOS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Regressao Logistica",
    "heuristica": "Heuristica",
}


@dataclass(slots=True)
class ResultadoClassificacao:
    classificacao: str          # "BOA" ou "RUIM"
    probabilidade: float        # probabilidade de a repeticao ser BOA
    modo: str                   # "modelo" ou "heuristico"

    @property
    def percentual(self) -> int:
        return int(round(self.probabilidade * 100))

    @property
    def rotulo_curto(self) -> str:
        return f"{self.classificacao} {self.percentual}%"


def _limitar(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    return max(minimo, min(maximo, float(valor)))


def probabilidade_heuristica(features: dict[str, float]) -> float:
    """Regra de bolso usada apenas quando nao ha modelo treinado.

    Cada criterio vira uma nota de 0 a 1 e o resultado e a media ponderada.
    Os pesos refletem o que mais indica execucao ruim: amplitude pequena,
    flexao incompleta, velocidade irregular, tronco balancando e cotovelo
    saindo do lugar.
    """
    amplitude = float(features.get("amplitude", 0.0))
    angulo_min = float(features.get("angulo_min", 180.0))
    vel_media = float(features.get("vel_media", 0.0))
    vel_std = float(features.get("vel_std", 0.0))
    tronco_std = float(features.get("inclinacao_tronco_std", 0.0))
    deriva = float(features.get("deriva_cotovelo", 0.0))
    intervalo = float(features.get("intervalo_anterior_s", 0.0))

    nota_rom = _limitar((amplitude - 60.0) / 55.0)
    nota_profundidade = _limitar((95.0 - angulo_min) / 35.0)
    coef_variacao = vel_std / max(vel_media, 1e-6)
    nota_ritmo = _limitar(1.0 - coef_variacao / 0.55)
    nota_tronco = _limitar(1.0 - tronco_std / 9.0)
    nota_cotovelo = _limitar(1.0 - deriva / 0.35)
    # Intervalo 0 = primeira repeticao da serie (sem penalidade).
    nota_cadencia = 1.0 if intervalo <= 0 else _limitar(1.0 - (intervalo - 4.0) / 6.0)

    return (
        0.28 * nota_rom
        + 0.24 * nota_profundidade
        + 0.20 * nota_ritmo
        + 0.15 * nota_tronco
        + 0.09 * nota_cotovelo
        + 0.04 * nota_cadencia
    )


class ClassificadorRepeticoes:
    """Fachada de inferencia: modelo treinado com fallback heuristico."""

    def __init__(self, caminho_modelo: Optional[str] = MODELO_PADRAO, usar_fallback: bool = True):
        self.caminho_modelo = caminho_modelo
        self.usar_fallback = usar_fallback
        self.modelo = None
        self._ordem_features: list[str] = list(FEATURE_NAMES)
        self.metadados: dict = {}
        self.modo = "heuristico"
        self.motivo_fallback = ""
        if caminho_modelo:
            self._carregar_modelo(caminho_modelo)

    # -- carregamento ------------------------------------------------------
    def _carregar_modelo(self, caminho: str) -> None:
        if not os.path.exists(caminho):
            self.motivo_fallback = f"modelo '{caminho}' nao encontrado"
            print(f"[ML] {self.motivo_fallback}. Usando heuristica.")
            return

        try:
            pacote = joblib.load(caminho)
        except Exception as erro:
            self.motivo_fallback = f"falha ao carregar '{caminho}': {erro}"
            print(f"[ML] {self.motivo_fallback}. Usando heuristica.")
            return

        if isinstance(pacote, dict) and "modelo" in pacote:
            self.modelo = pacote["modelo"]
            self._ordem_features = list(pacote.get("features") or FEATURE_NAMES)
            self.metadados = {chave: valor for chave, valor in pacote.items() if chave != "modelo"}
        else:  # compatibilidade com um estimador salvo "cru"
            self.modelo = pacote
            self._ordem_features = list(FEATURE_NAMES)
            self.metadados = {}

        faltando = set(FEATURE_NAMES) - set(self._ordem_features)
        if faltando:
            print(f"[ML] Modelo incompativel (faltam features: {sorted(faltando)}). Usando heuristica.")
            self.modelo = None
            self.motivo_fallback = "features incompativeis"
            return

        self.modo = "modelo"
        print(f"[ML] Modelo carregado: {self.descricao_modelo()} ({caminho})")

    # -- inferencia --------------------------------------------------------
    def classificar(self, features: dict[str, float]) -> ResultadoClassificacao:
        if self.modelo is not None:
            try:
                vetor = np.array([[float(features[nome]) for nome in self._ordem_features]])
                probabilidades = self.modelo.predict_proba(vetor)[0]
                classes = [str(classe) for classe in self.modelo.classes_]
                indice = classes.index("boa") if "boa" in classes else 0
                probabilidade = float(probabilidades[indice])
                modo = "modelo"
            except Exception as erro:
                if not self.usar_fallback:
                    raise
                print(f"[ML] Erro na inferencia ({erro}). Usando heuristica nesta repeticao.")
                probabilidade = probabilidade_heuristica(features)
                modo = "heuristico"
        else:
            if not self.usar_fallback:
                raise RuntimeError(
                    f"Sem modelo treinado disponivel ({self.motivo_fallback or 'modelo ausente'})."
                )
            probabilidade = probabilidade_heuristica(features)
            modo = "heuristico"

        return ResultadoClassificacao(
            classificacao="BOA" if probabilidade >= LIMIAR_BOA else "RUIM",
            probabilidade=probabilidade,
            modo=modo,
        )

    # -- metadados ---------------------------------------------------------
    @property
    def nome_modelo(self) -> str:
        if self.modo != "modelo":
            return NOMES_MODELOS["heuristica"]
        familia = str(self.metadados.get("modelo_escolhido", "random_forest"))
        return NOMES_MODELOS.get(familia, familia)

    @property
    def acuracia(self) -> Optional[float]:
        metricas = self.metadados.get("metricas") or {}
        valor = metricas.get("loo_accuracy")
        return float(valor) if valor is not None else None

    def descricao_modelo(self) -> str:
        if self.modo != "modelo":
            return "Heuristico (modelo ainda nao treinado)"
        if self.acuracia is not None:
            return f"{self.nome_modelo} - LOO {self.acuracia * 100:.0f}%"
        return self.nome_modelo

    def status(self) -> dict:
        return {
            "modo": self.modo,
            "nome_modelo": self.nome_modelo,
            "acuracia": self.acuracia,
            "descricao": self.descricao_modelo(),
            "motivo_fallback": self.motivo_fallback,
            "treinado_em": self.metadados.get("treinado_em"),
            "amostras": (self.metadados.get("metricas") or {}).get("n_amostras"),
        }


# ---------------------------------------------------------------------------
# Auto-teste
# ---------------------------------------------------------------------------
def _features_exemplo(tipo: str) -> dict[str, float]:
    if tipo == "boa":
        return {
            "angulo_min": 50.0, "angulo_max": 168.0, "amplitude": 118.0,
            "duracao_s": 2.1, "duracao_concentrica_s": 0.95,
            "vel_media": 108.0, "vel_max": 190.0, "vel_std": 12.0,
            "tempo_no_pico_s": 0.12, "inclinacao_tronco_media": 4.0,
            "inclinacao_tronco_std": 2.2, "abducao_ombro_media": 18.0,
            "deriva_cotovelo": 0.09, "confianca_media": 0.93,
            "intervalo_anterior_s": 2.6,
        }
    return {
        "angulo_min": 74.0, "angulo_max": 158.0, "amplitude": 84.0,
        "duracao_s": 3.5, "duracao_concentrica_s": 1.1,
        "vel_media": 66.0, "vel_max": 148.0, "vel_std": 36.0,
        "tempo_no_pico_s": 0.45, "inclinacao_tronco_media": 13.0,
        "inclinacao_tronco_std": 9.5, "abducao_ombro_media": 33.0,
        "deriva_cotovelo": 0.38, "confianca_media": 0.88,
        "intervalo_anterior_s": 5.2,
    }


def _auto_teste() -> None:
    print("=" * 70)
    print("classifier.py - auto-teste (modo heuristico)")
    print("=" * 70)

    classificador = ClassificadorRepeticoes(caminho_modelo=None)
    boa = classificador.classificar(_features_exemplo("boa"))
    ruim = classificador.classificar(_features_exemplo("fadiga"))

    print(f"Execucao boa    -> {boa.rotulo_curto} (p={boa.probabilidade:.2f}, {boa.modo})")
    print(f"Execucao fadiga -> {ruim.rotulo_curto} (p={ruim.probabilidade:.2f}, {ruim.modo})")

    assert boa.classificacao == "BOA", "heuristica deveria aprovar execucao boa"
    assert ruim.classificacao == "RUIM", "heuristica deveria reprovar execucao com fadiga"
    assert boa.probabilidade > ruim.probabilidade

    print("\nOK: heuristica separa BOA de FADIGA.")


if __name__ == "__main__":
    if os.environ.get("SMARTGYM_TESTAR_MODELO") == "1":
        classificador = ClassificadorRepeticoes()
        print(classificador.status())
        print(classificador.classificar(_features_exemplo("boa")))
        print(classificador.classificar(_features_exemplo("fadiga")))
    else:
        _auto_teste()
