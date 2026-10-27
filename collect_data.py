"""Coleta de dados rotulados para treinar o modelo (CP6).

Uso tipico (10 repeticoes boas + 10 com fadiga):

    python collect_data.py --classe boa    --reps 10
    python collect_data.py --classe fadiga --reps 10

Cada repeticao concluida vira uma linha de `dados_treino.csv`, com as mesmas
features usadas em tempo real por `index.py`.

Modo manual (rotula cada repeticao na hora):

    python collect_data.py --manual

Teclas no modo manual: [B] = boa, [R] = fadiga/ruim, [D] = descartar, [Q] = sair.

Observacoes:
- Nao e necessario exibir a garrafa durante a coleta: a classificacao usa
  apenas a pose (angulo do cotovelo + postura). A garrafa continua sendo exigida
  no treino normal (`index.py`).
- Grave as duas classes no mesmo enquadramento/luz do dia da demonstracao.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter
from datetime import datetime
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import (
    FEATURE_NAMES,
    RastreadorRosca,
    amostra_de_landmarks,
    extrair_features,
    formatar_features,
)

CAMINHO_CSV = "dados_treino.csv"
MODELO_POSE = "pose_landmarker_full.task"
CABECALHO = ["timestamp", "uid", "label", *FEATURE_NAMES]

ROTULOS_VALIDOS = {"boa", "fadiga"}


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def anexar_amostra(caminho: str, uid: str, label: str, features: dict[str, float]) -> None:
    """Acrescenta uma repeticao rotulada ao CSV, criando o cabecalho se preciso."""
    label = label.strip().lower()
    if label not in ROTULOS_VALIDOS:
        raise ValueError(f"Rotulo invalido: {label!r}. Use 'boa' ou 'fadiga'.")

    novo_arquivo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0
    with open(caminho, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CABECALHO)
        if novo_arquivo:
            escritor.writeheader()
        linha = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "uid": uid or "",
            "label": label,
        }
        linha.update({nome: round(float(features[nome]), 4) for nome in FEATURE_NAMES})
        escritor.writerow(linha)


def contar_rotulos(caminho: str) -> Counter:
    contagem: Counter = Counter()
    if not os.path.exists(caminho):
        return contagem
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            rotulo = (linha.get("label") or "").strip().lower()
            if rotulo:
                contagem[rotulo] += 1
    return contagem


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def _texto(frame, texto: str, linha: int, cor=(255, 255, 255), escala: float = 0.7,
           espessura: int = 2) -> None:
    cv2.putText(frame, texto, (12, linha), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura,
                cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------
def executar(args: argparse.Namespace) -> int:
    if not os.path.exists(MODELO_POSE):
        print(f"[Coleta] Arquivo de modelo ausente: {MODELO_POSE}")
        print("         Baixe o modelo de pose do MediaPipe e coloque na raiz do projeto.")
        return 1

    base_options = python.BaseOptions(model_asset_path=MODELO_POSE)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"[Coleta] Nao foi possivel abrir a camera {args.camera}.")
        return 1

    rastreador = RastreadorRosca()
    rotulo_automatico: Optional[str] = args.classe if not args.manual else None
    total_salvo = 0
    ultimo_inicio: Optional[float] = None
    pendente: Optional[dict[str, float]] = None
    pendente_info = ""
    mensagem = "Pose pronto. Comece a repeticao."
    mensagem_ate = time.perf_counter() + 3.0

    print("=" * 70)
    print(" SmartGym - coleta de dados (CP6)")
    print("=" * 70)
    print(f" Arquivo : {os.path.abspath(args.csv)}")
    if rotulo_automatico:
        print(f" Modo    : automatico | rotulo '{rotulo_automatico}' | alvo {args.reps} reps")
    else:
        print(" Modo    : manual | teclas [B] boa, [R] fadiga, [D] descartar, [Q] sair")
    print("=" * 70)

    t0 = time.perf_counter()
    ts_ms = 0

    try:
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            altura, largura, _ = frame.shape

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            ts_ms = max(ts_ms + 1, int((time.perf_counter() - t0) * 1000))
            resultado = detector.detect_for_video(mp_image, ts_ms)

            amostra = None
            if resultado.pose_landmarks:
                amostra = amostra_de_landmarks(
                    resultado.pose_landmarks[0], largura, altura, time.perf_counter() - t0
                )

            if amostra is not None:
                cv2.line(frame, tuple(map(int, amostra.ombro)), tuple(map(int, amostra.cotovelo)),
                         (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(frame, tuple(map(int, amostra.cotovelo)), tuple(map(int, amostra.pulso)),
                         (255, 255, 255), 3, cv2.LINE_AA)

                repeticao = rastreador.atualizar(amostra, garrafa_visivel=True)
                if repeticao is not None:
                    intervalo = 0.0
                    if ultimo_inicio is not None:
                        intervalo = max(0.0, repeticao.amostras[0].t - ultimo_inicio)
                    features = extrair_features(repeticao, intervalo_anterior_s=intervalo)

                    if rotulo_automatico:
                        anexar_amostra(args.csv, args.uid, rotulo_automatico, features)
                        total_salvo += 1
                        mensagem = f"Salva #{total_salvo}: {rotulo_automatico.upper()} | {formatar_features(features)}"
                        mensagem_ate = time.perf_counter() + 3.0
                        print(f"[Coleta] #{total_salvo:02d} {rotulo_automatico:6s} "
                              f"| {formatar_features(features)}")
                        if args.reps and total_salvo >= args.reps:
                            break
                    else:
                        pendente = features
                        pendente_info = formatar_features(features)
                        mensagem = "Repeticao detectada: [B] BOA  [R] FADIGA  [D] descartar"
                        mensagem_ate = time.perf_counter() + 8.0

                    ultimo_inicio = repeticao.amostras[0].t

            # --- HUD -------------------------------------------------------
            cor = (0, 255, 0) if amostra is not None else (0, 0, 255)
            _texto(frame, f"COLETA DE DADOS - alvo: {args.reps or 'livre'} reps", 30, (255, 255, 0), 0.7)
            _texto(frame, f"Salvas nesta sessao: {total_salvo}", 62, (255, 255, 255), 0.65)
            _texto(frame, f"Fase: {rastreador.estagio}", 94, cor, 0.6)
            if amostra is not None:
                _texto(frame, f"Angulo do cotovelo: {amostra.angulo:.1f} deg", 124, (0, 255, 255), 0.6)

            if pendente is not None:
                _texto(frame, pendente_info, 158, (0, 200, 255), 0.55)
            if time.perf_counter() < mensagem_ate:
                _texto(frame, mensagem, altura - 30, (0, 255, 0), 0.6)
            else:
                _texto(frame, "Tecla Q para encerrar", altura - 30, (180, 180, 180), 0.55, 1)

            contagem = contar_rotulos(args.csv)
            _texto(frame, f"CSV: boa={contagem.get('boa', 0)}  fadiga={contagem.get('fadiga', 0)}",
                   158 if pendente is None else 186, (200, 200, 200), 0.55)

            cv2.imshow("SmartGym - coleta (CP6)", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            if pendente is not None and tecla in (ord("b"), ord("r")):
                rotulo = "boa" if tecla == ord("b") else "fadiga"
                anexar_amostra(args.csv, args.uid, rotulo, pendente)
                total_salvo += 1
                print(f"[Coleta] #{total_salvo:02d} {rotulo:6s} (manual) | {pendente_info}")
                pendente = None
                mensagem = f"Salva como {rotulo.upper()}."
                mensagem_ate = time.perf_counter() + 2.0
                if args.reps and total_salvo >= args.reps:
                    break
            elif pendente is not None and tecla == ord("d"):
                print("[Coleta] Repeticao descartada.")
                pendente = None
                mensagem = "Repeticao descartada."
                mensagem_ate = time.perf_counter() + 2.0
    finally:
        camera.release()
        cv2.destroyAllWindows()

    contagem = contar_rotulos(args.csv)
    print("-" * 70)
    print(f" Sessao encerrada. Salvas agora: {total_salvo}")
    print(f" Total no CSV ({args.csv}): boa={contagem.get('boa', 0)} | "
          f"fadiga={contagem.get('fadiga', 0)}")
    if contagem.get("boa", 0) < 10 or contagem.get("fadiga", 0) < 10:
        print(" ATENCAO: o ideal e ter pelo menos 10 repeticoes de cada classe.")
    print(" Proximo passo: python train_model.py")
    print("-" * 70)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Coleta de repeticoes rotuladas para o modelo.")
    parser.add_argument("--classe", choices=sorted(ROTULOS_VALIDOS), default=None,
                        help="Rotulo automatico das repeticoes (boa/fadiga).")
    parser.add_argument("--manual", action="store_true",
                        help="Rotula cada repeticao no teclado (B/R/D).")
    parser.add_argument("--reps", type=int, default=10,
                        help="Quantas repeticoes salvar nesta sessao (0 = sem limite).")
    parser.add_argument("--aluno", dest="uid", default="", help="UID do aluno (opcional, vai para o CSV).")
    parser.add_argument("--csv", default=CAMINHO_CSV, help="Arquivo CSV de saida.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam.")
    args = parser.parse_args(argv)

    if args.manual and args.classe:
        parser.error("Use --manual OU --classe, nao os dois.")
    if not args.manual and not args.classe:
        parser.error("Informe --classe boa/fadiga para rotulagem automatica ou --manual.")

    return executar(args)


if __name__ == "__main__":
    raise SystemExit(main())
