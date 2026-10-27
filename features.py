"""Extracao de caracteristicas (features) por repeticao - Smart Gym / CP6.

Este modulo NAO depende de camera, MediaPipe ou scikit-learn. Ele recebe os
landmarks ja convertidos para pixels (AmostraQuadro) e devolve um dicionario
numerico que descreve a qualidade da execucao de UMA repeticao de rosca direita.

A mesma funcao `extrair_features` e usada em dois momentos:

1. Coleta de dados (`collect_data.py`) -> vira uma linha do CSV de treino.
2. Inferencia em tempo real (`index.py`) -> entra no modelo ja treinado.

Manter o mesmo caminho de codigo nos dois lados evita o classico problema de
"treinar com uma feature e inferir com outra".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Parametros do movimento (rosca direita)
# ---------------------------------------------------------------------------
ANGULO_EXTENDIDO = 160.0     # cotovelo praticamente esticado (inicio/fim da rep)
ANGULO_CONTRAIDO = 60.0      # cotovelo totalmente flexionado (pico da rep)
FRAMES_EXTENSAO = 8          # quantos quadros no topo do movimento entram na rep
MIN_FRAMES_REP = 5           # rep muito curta = ruido de deteccao
MIN_AMPLITUDE_REP = 20.0     # graus; abaixo disso nao e uma repeticao de verdade
LIMIAR_CONFIANCA = 0.35      # visibilidade minima dos landmarks do braco
TOLERANCIA_PICO = 12.0       # graus acima do angulo minimo = "no pico"

# Ordem canonica das features. O modelo salvo guarda esta lista junto, mas a
# ordem aqui e a referencia para o CSV e para o vetor de inferencia.
FEATURE_NAMES = [
    "angulo_min",                 # profundidade da flexao (°)
    "angulo_max",                 # extensao alcancada (°)
    "amplitude",                  # amplitude real do movimento (°)
    "duracao_s",                  # duracao total da repeticao (s)
    "duracao_concentrica_s",      # tempo do inicio ate o pico de flexao (s)
    "vel_media",                  # velocidade angular media (°/s)
    "vel_max",                    # velocidade angular maxima (°/s)
    "vel_std",                    # irregularidade da velocidade (°/s)
    "tempo_no_pico_s",            # tempo parado na contracao (s)
    "inclinacao_tronco_media",    # inclinacao media do tronco (°)
    "inclinacao_tronco_std",      # quanto o tronco balanca (°)
    "abducao_ombro_media",        # abertura do ombro durante a rep (°)
    "deriva_cotovelo",            # deslocamento do cotovelo (normalizado)
    "confianca_media",            # visibilidade media dos landmarks (0-1)
    "intervalo_anterior_s",       # tempo desde o inicio da rep anterior (s)
]

DESCRICOES_FEATURES = {
    "angulo_min": "Menor angulo do cotovelo (quanto mais baixo, mais completa a flexao).",
    "angulo_max": "Maior angulo do cotovelo (extensao alcancada).",
    "amplitude": "Diferenca entre extensao e flexao maxima; encurta com fadiga.",
    "duracao_s": "Tempo total da repeticao; tende a crescer com fadiga.",
    "duracao_concentrica_s": "Tempo da fase de subida; fadiga atrasa a contracao.",
    "vel_media": "Velocidade angular media do movimento.",
    "vel_max": "Pico de velocidade angular.",
    "vel_std": "Variacao da velocidade: movimento tremido/irregular aumenta este valor.",
    "tempo_no_pico_s": "Quanto tempo o aluno permanece na contracao maxima.",
    "inclinacao_tronco_media": "Inclinacao media do tronco (roubo de movimento).",
    "inclinacao_tronco_std": "Oscilacao do tronco durante a repeticao (balanco).",
    "abducao_ombro_media": "Abertura do ombro; compensacao tipica de fadiga.",
    "deriva_cotovelo": "Quanto o cotovelo 'passeia' em relacao ao ombro.",
    "confianca_media": "Qualidade da deteccao de pose na repeticao.",
    "intervalo_anterior_s": "Intervalo desde a repeticao anterior (0 na primeira).",
}


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AmostraQuadro:
    """Um quadro de pose ja normalizado para pixels."""

    t: float
    angulo: float
    ombro: tuple[float, float]
    cotovelo: tuple[float, float]
    pulso: tuple[float, float]
    quadril: tuple[float, float]
    largura_ombros: float
    confianca: float


@dataclass(slots=True)
class RepeticaoConcluida:
    """Repeticao fechada (voltou a extensao) pronta para virar features."""

    amostras: list[AmostraQuadro]
    garrafa_visivel: bool
    angulo_min: float
    angulo_max: float
    duracao_s: float

    @property
    def amplitude(self) -> float:
        return self.angulo_max - self.angulo_min

    @property
    def confianca_media(self) -> float:
        if not self.amostras:
            return 0.0
        return mean(a.confianca for a in self.amostras)


# ---------------------------------------------------------------------------
# Geometria e leitura dos landmarks
# ---------------------------------------------------------------------------
def calcular_angulo(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Angulo em `b` formado pelos segmentos b->a e b->c (em graus)."""
    radianos = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angulo = abs(math.degrees(radianos))
    if angulo > 180.0:
        angulo = 360.0 - angulo
    return angulo


def _visibilidade(landmark) -> float:
    valor = getattr(landmark, "visibility", None)
    return 1.0 if valor is None else float(valor)


def amostra_de_landmarks(landmarks, largura: int, altura: int, t: float) -> Optional[AmostraQuadro]:
    """Converte landmarks do MediaPipe em uma AmostraQuadro.

    Escolhe o braco com melhor visibilidade (esquerdo por padrao) e devolve
    None quando a pose esta ruim demais para gerar dados confiaveis.

    Indices MediaPipe: 11/12 ombros, 13/14 cotovelos, 15/16 pulsos, 23/24 quadris.
    """
    if landmarks is None:
        return None

    def ponto(indice: int) -> tuple[float, float]:
        lm = landmarks[indice]
        return (float(lm.x) * largura, float(lm.y) * altura)

    vis_esq = mean(_visibilidade(landmarks[i]) for i in (11, 13, 15))
    vis_dir = mean(_visibilidade(landmarks[i]) for i in (12, 14, 16))

    if vis_esq >= LIMIAR_CONFIANCA:
        indices = (11, 13, 15, 23, 24)
    elif vis_dir >= LIMIAR_CONFIANCA:
        indices = (12, 14, 16, 24, 23)
    else:
        return None

    ombro_i, cotovelo_i, pulso_i, quadril_i, _outro_quadril = indices
    confianca = mean(_visibilidade(landmarks[i]) for i in indices[:4])
    if confianca < LIMIAR_CONFIANCA:
        return None

    ombro = ponto(ombro_i)
    cotovelo = ponto(cotovelo_i)
    pulso = ponto(pulso_i)
    quadril = ponto(quadril_i)

    ombro_esq = ponto(11)
    ombro_dir = ponto(12)
    largura_ombros = math.hypot(ombro_dir[0] - ombro_esq[0], ombro_dir[1] - ombro_esq[1])
    if largura_ombros < 1.0:
        return None

    return AmostraQuadro(
        t=t,
        angulo=calcular_angulo(ombro, cotovelo, pulso),
        ombro=ombro,
        cotovelo=cotovelo,
        pulso=pulso,
        quadril=quadril,
        largura_ombros=largura_ombros,
        confianca=confianca,
    )


# ---------------------------------------------------------------------------
# Maquina de estados: fecha uma repeticao completa
# ---------------------------------------------------------------------------
class RastreadorRosca:
    """Detecta repeticoes completas (extensao -> flexao -> extensao).

    A repeticao so e emitida quando o cotovelo volta a ficar estendido depois de
    ter passado pela contracao. Isso garante que as features representem o ciclo
    inteiro e evita contar movimento parcial no fundo do movimento.
    """

    def __init__(
        self,
        angulo_extendido: float = ANGULO_EXTENDIDO,
        angulo_contraido: float = ANGULO_CONTRAIDO,
        frames_extensao: int = FRAMES_EXTENSAO,
        min_frames: int = MIN_FRAMES_REP,
        min_amplitude: float = MIN_AMPLITUDE_REP,
    ) -> None:
        self.angulo_extendido = float(angulo_extendido)
        self.angulo_contraido = float(angulo_contraido)
        self.frames_extensao = max(1, int(frames_extensao))
        self.min_frames = max(2, int(min_frames))
        self.min_amplitude = float(min_amplitude)

        self.estagio = "EXTENDIDO"
        self._extensao: list[AmostraQuadro] = []
        self._ativas: list[AmostraQuadro] | None = None
        self._viu_garrafa = False

    @property
    def em_movimento(self) -> bool:
        return self.estagio == "CONTRAIDO"

    def atualizar(self, amostra: AmostraQuadro, garrafa_visivel: bool = False) -> Optional[RepeticaoConcluida]:
        """Consome um quadro e devolve a repeticao quando ela e concluida."""
        angulo = amostra.angulo

        if angulo >= self.angulo_extendido:
            if self.estagio == "CONTRAIDO" and self._ativas is not None:
                amostras = self._ativas + [amostra]
                viu_garrafa = self._viu_garrafa or garrafa_visivel
                self._resetar(primeira_amostra=amostra)
                return self._montar_repeticao(amostras, viu_garrafa)

            self._ativas = None
            self._viu_garrafa = bool(garrafa_visivel)
            self._extensao.append(amostra)
            if len(self._extensao) > self.frames_extensao:
                self._extensao.pop(0)
            return None

        if angulo < self.angulo_contraido:
            if self.estagio == "EXTENDIDO":
                self.estagio = "CONTRAIDO"
                self._ativas = self._extensao + [amostra]
                self._viu_garrafa = bool(garrafa_visivel)
                self._extensao = []
            else:
                self._ativas = (self._ativas or []) + [amostra]
                self._viu_garrafa = self._viu_garrafa or garrafa_visivel
            return None

        # Zona intermediaria: acumula o caminho do movimento.
        if self.estagio == "CONTRAIDO":
            self._ativas = (self._ativas or []) + [amostra]
            self._viu_garrafa = self._viu_garrafa or garrafa_visivel
        else:
            self._extensao.append(amostra)
            if len(self._extensao) > self.frames_extensao:
                self._extensao.pop(0)
        return None

    # -- internos ----------------------------------------------------------
    def _resetar(self, primeira_amostra: AmostraQuadro) -> None:
        self.estagio = "EXTENDIDO"
        self._ativas = None
        self._viu_garrafa = False
        self._extensao = [primeira_amostra]

    def _montar_repeticao(
        self, amostras: list[AmostraQuadro], viu_garrafa: bool
    ) -> Optional[RepeticaoConcluida]:
        if len(amostras) < self.min_frames:
            return None
        angulos = [a.angulo for a in amostras]
        angulo_min = min(angulos)
        angulo_max = max(angulos)
        if (angulo_max - angulo_min) < self.min_amplitude:
            return None
        duracao = max(amostras[-1].t - amostras[0].t, 0.0)
        return RepeticaoConcluida(
            amostras=amostras,
            garrafa_visivel=bool(viu_garrafa),
            angulo_min=angulo_min,
            angulo_max=angulo_max,
            duracao_s=duracao,
        )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def extrair_features(rep: RepeticaoConcluida, intervalo_anterior_s: float = 0.0) -> dict[str, float]:
    """Transforma uma repeticao completa no dicionario de features do modelo."""
    amostras = rep.amostras
    if len(amostras) < 3:
        raise ValueError("Repeticao precisa de pelo menos 3 amostras para gerar features.")

    tempos = [a.t for a in amostras]
    angulos = [a.angulo for a in amostras]
    angulo_min = min(angulos)
    angulo_max = max(angulos)
    amplitude = angulo_max - angulo_min
    duracao = max(tempos[-1] - tempos[0], 1e-6)
    indice_pico = angulos.index(angulo_min)
    duracao_concentrica = max(tempos[indice_pico] - tempos[0], 0.0)

    velocidades: list[float] = []
    tempo_no_pico = 0.0
    for i in range(1, len(amostras)):
        dt = tempos[i] - tempos[i - 1]
        if dt <= 1e-4:
            continue
        velocidades.append(abs(angulos[i] - angulos[i - 1]) / dt)
        if angulos[i] <= angulo_min + TOLERANCIA_PICO and angulos[i - 1] <= angulo_min + TOLERANCIA_PICO:
            tempo_no_pico += dt

    if velocidades:
        vel_media = mean(velocidades)
        vel_max = max(velocidades)
        vel_std = pstdev(velocidades) if len(velocidades) > 1 else 0.0
    else:
        vel_media = vel_max = vel_std = 0.0

    inclinacoes = []
    abducoes = []
    for a in amostras:
        dx = a.ombro[0] - a.quadril[0]
        dy = a.quadril[1] - a.ombro[1]
        inclinacoes.append(abs(math.degrees(math.atan2(dx, dy))))
        abducoes.append(calcular_angulo(a.quadril, a.ombro, a.cotovelo))

    larguras = [a.largura_ombros for a in amostras if a.largura_ombros > 0]
    largura_ref = mean(larguras) if larguras else 1.0
    desvios_x = [(a.cotovelo[0] - a.ombro[0]) / largura_ref for a in amostras]
    desvios_y = [(a.cotovelo[1] - a.ombro[1]) / largura_ref for a in amostras]
    deriva_cotovelo = math.hypot(
        pstdev(desvios_x) if len(desvios_x) > 1 else 0.0,
        pstdev(desvios_y) if len(desvios_y) > 1 else 0.0,
    )

    return {
        "angulo_min": angulo_min,
        "angulo_max": angulo_max,
        "amplitude": amplitude,
        "duracao_s": duracao,
        "duracao_concentrica_s": duracao_concentrica,
        "vel_media": vel_media,
        "vel_max": vel_max,
        "vel_std": vel_std,
        "tempo_no_pico_s": tempo_no_pico,
        "inclinacao_tronco_media": mean(inclinacoes),
        "inclinacao_tronco_std": pstdev(inclinacoes) if len(inclinacoes) > 1 else 0.0,
        "abducao_ombro_media": mean(abducoes),
        "deriva_cotovelo": deriva_cotovelo,
        "confianca_media": rep.confianca_media,
        "intervalo_anterior_s": float(max(intervalo_anterior_s, 0.0)),
    }


def formatar_features(features: dict[str, float]) -> str:
    """Resumo curto para console/overlay."""
    return (
        f"ROM {features['amplitude']:.0f}deg | "
        f"min {features['angulo_min']:.0f}deg | "
        f"{features['duracao_s']:.2f}s | "
        f"vel {features['vel_media']:.0f}deg/s | "
        f"tronco {features['inclinacao_tronco_std']:.1f}deg"
    )


# ---------------------------------------------------------------------------
# Auto-teste (nao depende de camera)
# ---------------------------------------------------------------------------
def _amostra_sintetica(t: float, angulo: float, inclinacao: float = 0.0, deriva: float = 0.0,
                       confianca: float = 0.9) -> AmostraQuadro:
    ombro = (200.0, 120.0)
    quadril = (200.0 + math.sin(math.radians(inclinacao)) * 160.0,
               120.0 + math.cos(math.radians(inclinacao)) * 160.0)
    cotovelo = (200.0 + deriva, 220.0)
    pulso = (200.0 + deriva + math.sin(math.radians(180 - angulo)) * 70.0,
             220.0 + math.cos(math.radians(180 - angulo)) * 70.0)
    return AmostraQuadro(
        t=t,
        angulo=angulo,
        ombro=ombro,
        cotovelo=cotovelo,
        pulso=pulso,
        quadril=quadril,
        largura_ombros=180.0,
        confianca=confianca,
    )


def _gerar_rep_sintetica(n: int, duracao: float, minima: float, maxima: float,
                         ruido: float = 0.0, inclinacao_base: float = 0.0,
                         inclinacao_oscilacao: float = 0.0, deriva: float = 0.0) -> RepeticaoConcluida:
    amostras = []
    for i in range(n):
        fracao = i / (n - 1)
        # 1 no inicio/fim (extensao maxima) e 0 no meio (flexao maxima).
        curva = 0.5 + 0.5 * math.cos(2.0 * math.pi * fracao)
        angulo = minima + (maxima - minima) * curva
        angulo += math.sin(i * 1.7) * ruido
        inclinacao = inclinacao_base + math.sin(2.0 * math.pi * fracao) * inclinacao_oscilacao
        amostras.append(_amostra_sintetica(
            t=fracao * duracao,
            angulo=max(0.0, min(180.0, angulo)),
            inclinacao=inclinacao,
            deriva=deriva * fracao,
        ))
    angulos = [a.angulo for a in amostras]
    return RepeticaoConcluida(
        amostras=amostras,
        garrafa_visivel=True,
        angulo_min=min(angulos),
        angulo_max=max(angulos),
        duracao_s=amostras[-1].t - amostras[0].t,
    )


def _auto_teste() -> None:
    print("=" * 70)
    print("features.py - auto-teste")
    print("=" * 70)

    boa = _gerar_rep_sintetica(60, 2.0, minima=50, maxima=160)
    fadiga = _gerar_rep_sintetica(
        100, 3.4, minima=78, maxima=158,
        ruido=3.0, inclinacao_base=10.0, inclinacao_oscilacao=8.0, deriva=45.0,
    )

    feats_boa = extrair_features(boa, intervalo_anterior_s=2.4)
    feats_fadiga = extrair_features(fadiga, intervalo_anterior_s=4.8)

    print("\n[BOA]   ", formatar_features(feats_boa))
    print("[FADIGA]", formatar_features(feats_fadiga))

    assert set(feats_boa) == set(FEATURE_NAMES), "features fora do contrato"
    assert feats_boa["amplitude"] > feats_fadiga["amplitude"], "amplitude deveria cair na fadiga"
    assert feats_fadiga["vel_std"] > feats_boa["vel_std"], "fadiga deveria ser mais irregular"
    assert feats_fadiga["inclinacao_tronco_std"] > feats_boa["inclinacao_tronco_std"], \
        "fadiga deveria balancar mais o tronco"

    # Rastreador: alimenta um ciclo completo e espera exatamente 1 repeticao.
    rastreador = RastreadorRosca()
    emitidas = 0
    for a in boa.amostras:
        if rastreador.atualizar(a, garrafa_visivel=True) is not None:
            emitidas += 1
    assert emitidas == 1, f"esperava 1 repeticao, obteve {emitidas}"
    assert rastreador.estagio == "EXTENDIDO"

    # Movimento incompleto nao conta.
    rastreador = RastreadorRosca()
    emitidas = 0
    for a in _gerar_rep_sintetica(40, 1.2, minima=110, maxima=150).amostras:
        if rastreador.atualizar(a, garrafa_visivel=True) is not None:
            emitidas += 1
    assert emitidas == 0, "movimento parcial nao deveria contar repeticao"

    print("\nOK: features e maquina de estados consistentes.")
    print("Features:", ", ".join(FEATURE_NAMES))


if __name__ == "__main__":
    _auto_teste()
