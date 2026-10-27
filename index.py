"""Smart Gym - Estacao 01 (CP6 / IoB).

Fluxo completo:

    RFID (Arduino) -> valida no SQLite -> webcam + MediaPipe/YOLO
        -> maquina de estados fecha a repeticao
        -> features.py extrai as caracteristicas
        -> classifier.py classifica BOA x RUIM (fadiga)
        -> SQLite + MQTT + dashboard com Score de Eficiencia

Execucao normal (com camera, RFID e MQTT):

    python index.py

Demonstracao sem camera (gera repeticoes simuladas, grava no banco, publica
MQTT e abre o dashboard):

    python index.py --demo
    python index.py --demo --sem-mqtt --png resultado_treino_demo.png
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import time
import tkinter as tk
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

import db
from classifier import ClassificadorRepeticoes
from dashboard import DashboardTreino, RepResumo, ResumoTreino
from features import (
    RastreadorRosca,
    amostra_de_landmarks,
    extrair_features,
)
from mqtt_client import (
    ClienteMQTT,
    MQTT_HOST,
    MQTT_PORT,
    MQTT_TOPIC_BASE,
    publicar_fim_treino,
    publicar_repeticao,
)

# ---------------------------------------------------------------------------
# Configuracao de hardware
# ---------------------------------------------------------------------------
PORTA_COM = "COM9"
BAUD_RATE = 9600
TIMEOUT_SERIAL = 1
INTERVALO_LEITURA_SERIAL_MS = 150
INTERVALO_RECONEXAO_SERIAL = 10

CLASSE_GARRAFA = 39       # classe "bottle" no YOLOv8 (COCO)
CONF_MINIMA = 0.25

MODELO_POSE = "pose_landmarker_full.task"
MODELO_YOLO = "yolov8s.pt"
CAMERA = 0

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
conn: Optional[sqlite3.Connection] = None
detector_pose = None
modelo_yolo = None
classificador: Optional[ClassificadorRepeticoes] = None
mqtt_cliente: Optional[ClienteMQTT] = None

arduino = None
buffer_serial = ""
ultima_tentativa_serial = 0.0

root: Optional[tk.Tk] = None
label_status: Optional[tk.Label] = None
label_mqtt: Optional[tk.Label] = None
label_modelo: Optional[tk.Label] = None
entry_id: Optional[tk.Entry] = None
sessao_ativa = False


# ---------------------------------------------------------------------------
# Serial (RFID)
# ---------------------------------------------------------------------------
def tentar_conectar_serial(forcar: bool = False) -> bool:
    """Abre a porta serial. Retorna True se a porta esta disponivel."""
    global arduino, ultima_tentativa_serial

    if arduino is not None and arduino.is_open:
        return True

    agora = time.time()
    if not forcar and (agora - ultima_tentativa_serial) < INTERVALO_RECONEXAO_SERIAL:
        return False
    ultima_tentativa_serial = agora

    try:
        import serial  # importado sob demanda para nao travar quem so usa --demo

        arduino = serial.Serial(PORTA_COM, BAUD_RATE, timeout=TIMEOUT_SERIAL)
        time.sleep(2)
        print(f"[RFID] Conectado na porta {PORTA_COM}.")
        return True
    except Exception as erro:
        print(f"[RFID] Nao foi possivel abrir {PORTA_COM}: {erro}")
        arduino = None
        return False


# ---------------------------------------------------------------------------
# HUD da camera
# ---------------------------------------------------------------------------
def _texto(frame, texto: str, x: int, y: int, cor=(255, 255, 255),
           escala: float = 0.7, espessura: int = 2) -> None:
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura,
                cv2.LINE_AA)


def _desenhar_braco(frame, amostra) -> None:
    cv2.line(frame, tuple(map(int, amostra.ombro)), tuple(map(int, amostra.cotovelo)),
             (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, tuple(map(int, amostra.cotovelo)), tuple(map(int, amostra.pulso)),
             (255, 255, 255), 3, cv2.LINE_AA)


def _desenhar_hud(
    frame,
    *,
    nome: str,
    exercicio: str,
    meta: int,
    total: int,
    boas: int,
    garrafa_visivel: bool,
    estagio: str,
    angulo: float,
    ultimas: list[tuple[int, str]],
    mensagem: str,
) -> None:
    altura, largura, _ = frame.shape
    score_parcial = (100.0 * boas / total) if total else 0.0

    _texto(frame, f"Aluno: {nome} | {exercicio}", 12, 30, (255, 255, 255), 0.62)
    _texto(frame, f"CONTAGEM: {total}/{meta}", 12, 66, (0, 255, 0), 0.85, 3)

    cor_garrafa = (0, 255, 0) if garrafa_visivel else (0, 0, 255)
    status_garrafa = "Garrafa: VISIVEL" if garrafa_visivel else "Garrafa: AUSENTE"
    _texto(frame, status_garrafa, 12, 96, cor_garrafa, 0.62)

    _texto(frame, f"Fase: {estagio}", 12, 126, (255, 255, 0), 0.62)
    _texto(frame, f"Angulo do cotovelo: {angulo:.1f} deg", 12, 156, (0, 255, 255), 0.6)
    _texto(frame, f"Score parcial: {score_parcial:.0f}%  ({boas} boas / {total})",
           12, 186, (0, 200, 255), 0.6)

    if ultimas:
        _texto(frame, "Ultimas repeticoes:", max(12, largura - 330), 30, (200, 200, 200), 0.55, 1)
        for indice, (numero, rotulo) in enumerate(ultimas):
            cor = (0, 255, 0) if rotulo.startswith("BOA") else (0, 0, 255)
            _texto(frame, f"rep {numero}: {rotulo}", max(12, largura - 330),
                   58 + indice * 26, cor, 0.55, 1)

    if mensagem:
        tamanho = cv2.getTextSize(mensagem, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        _texto(frame, mensagem, max(12, (largura - tamanho[0]) // 2), altura - 24,
               (0, 255, 255), 0.7)


# ---------------------------------------------------------------------------
# Sessao de treino com camera
# ---------------------------------------------------------------------------
def iniciar_exercicio(nome: str, exercicio: str, meta: int, uid: str) -> ResumoTreino:
    """Executa o treino com visao computacional e devolve o resumo classificado."""
    if classificador is None or detector_pose is None or modelo_yolo is None or conn is None:
        raise RuntimeError("Recursos de visao/ML nao inicializados.")

    meta = max(1, int(meta))
    id_treino = db.criar_treino(conn, uid, exercicio, meta, classificador.descricao_modelo())
    camera = cv2.VideoCapture(CAMERA)
    if not camera.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera {CAMERA}.")

    rastreador = RastreadorRosca()
    t0 = time.perf_counter()
    ts_ms = 0
    reps: list[RepResumo] = []
    ultimas: list[tuple[int, str]] = []
    boas = 0
    inicio_ultima_rep: Optional[float] = None
    mensagem = "Faca a primeira repeticao"
    mensagem_ate = t0 + 4.0
    print(f"[Treino] {exercicio} liberado para {nome}: {meta} repeticoes. "
          f"Modelo: {classificador.descricao_modelo()}")

    try:
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[Treino] Falha ao ler a camera.")
                break

            frame = cv2.flip(frame, 1)
            altura, largura, _ = frame.shape

            # --- YOLO: garrafa -------------------------------------------------
            garrafa_agora = False
            for resultado_yolo in modelo_yolo(
                frame, classes=[CLASSE_GARRAFA], verbose=False, conf=CONF_MINIMA
            ):
                if len(resultado_yolo.boxes) > 0:
                    garrafa_agora = True
                    frame = resultado_yolo.plot()

            # --- MediaPipe: pose ----------------------------------------------
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            ts_ms = max(ts_ms + 1, int((time.perf_counter() - t0) * 1000))
            resultado_pose = detector_pose.detect_for_video(mp_image, ts_ms)

            amostra = None
            if resultado_pose.pose_landmarks:
                amostra = amostra_de_landmarks(
                    resultado_pose.pose_landmarks[0], largura, altura,
                    time.perf_counter() - t0,
                )

            if amostra is not None:
                _desenhar_braco(frame, amostra)
                repeticao = rastreador.atualizar(amostra, garrafa_visivel=garrafa_agora)

                if repeticao is not None:
                    if not repeticao.garrafa_visivel:
                        mensagem = "GARRAFA AUSENTE: repeticao descartada!"
                        mensagem_ate = time.perf_counter() + 3.0
                        print("[Treino] Repeticao descartada: garrafa nao visivel.")
                    else:
                        intervalo = 0.0
                        if inicio_ultima_rep is not None:
                            intervalo = max(0.0, repeticao.amostras[0].t - inicio_ultima_rep)

                        features = extrair_features(repeticao, intervalo_anterior_s=intervalo)
                        resultado = classificador.classificar(features)
                        numero = len(reps) + 1
                        boas += 1 if resultado.classificacao == "BOA" else 0
                        ruins = numero - boas
                        score = 100.0 * boas / numero

                        db.registrar_repeticao(
                            conn, id_treino, numero, resultado.classificacao,
                            resultado.probabilidade, features,
                        )
                        publicar_repeticao(
                            mqtt_cliente,
                            aluno=nome, uid=uid, exercicio=exercicio, meta=meta,
                            numero=numero, fase=rastreador.estagio,
                            angulo=repeticao.angulo_min, garrafa_visivel=True,
                            classificacao=resultado.classificacao,
                            probabilidade=resultado.probabilidade,
                            boas=boas, ruins=ruins, score=score,
                        )

                        reps.append(RepResumo(
                            numero=numero,
                            classificacao=resultado.classificacao,
                            probabilidade=resultado.probabilidade,
                            angulo_min=repeticao.angulo_min,
                            angulo_max=repeticao.angulo_max,
                        ))
                        ultimas = (ultimas + [(numero, resultado.rotulo_curto)])[-3:]
                        inicio_ultima_rep = repeticao.amostras[0].t
                        mensagem = f"REP {numero}: {resultado.classificacao} ({resultado.percentual}%)"
                        mensagem_ate = time.perf_counter() + 2.5

                        print(
                            f"[Treino] Rep {numero}/{meta} -> {resultado.classificacao} "
                            f"({resultado.percentual}%) | ROM {features['amplitude']:.0f} deg | "
                            f"score {score:.1f}%"
                        )
                        if len(reps) >= meta:
                            break

            _desenhar_hud(
                frame,
                nome=nome, exercicio=exercicio, meta=meta, total=len(reps), boas=boas,
                garrafa_visivel=garrafa_agora, estagio=rastreador.estagio,
                angulo=amostra.angulo if amostra is not None else 0.0,
                ultimas=ultimas,
                mensagem=mensagem if time.perf_counter() < mensagem_ate else "",
            )

            cv2.imshow("SmartGym", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Treino] Encerrado pelo usuario (q).")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    duracao = time.perf_counter() - t0
    total = len(reps)
    boas = sum(1 for rep in reps if rep.classificacao == "BOA")
    ruins = total - boas
    score_final = (100.0 * boas / total) if total else 0.0

    db.finalizar_treino(conn, id_treino, boas, ruins, score_final)
    publicar_fim_treino(
        mqtt_cliente,
        aluno=nome, uid=uid, exercicio=exercicio, meta=meta,
        boas=boas, ruins=ruins, score=score_final, duracao_s=duracao,
        modelo=classificador.descricao_modelo(),
    )

    print(f"[Treino] Fim: {boas} boas / {ruins} com fadiga -> score {score_final:.1f}%")

    return ResumoTreino(
        aluno=nome,
        uid=uid,
        exercicio=exercicio,
        meta=meta,
        reps=reps,
        modelo=classificador.descricao_modelo(),
        duracao_s=duracao,
        id_treino=id_treino,
        origem="camera",
    )


# ---------------------------------------------------------------------------
# Login / RFID
# ---------------------------------------------------------------------------
def _texto_status(texto: str, cor: str = "black") -> None:
    if label_status is not None:
        label_status.config(text=texto, fg=cor)


def abrir_sessao(uid: str, nome: str, exercicio: str, meta: int) -> None:
    """Registra o acesso, executa o treino e mostra o dashboard final."""
    global sessao_ativa

    if root is None or entry_id is None or conn is None:
        return
    if sessao_ativa:
        _texto_status("Aguarde: ja existe um treino em andamento.", "orange")
        return

    sessao_ativa = True
    try:
        try:
            db.registrar_acesso(conn, uid)
        except Exception as erro:
            print(f"[DB] Falha ao registrar acesso: {erro}")

        _texto_status(f"Treino ativo - {nome}", "green")
        root.withdraw()
        resumo: Optional[ResumoTreino] = None

        try:
            resumo = iniciar_exercicio(nome, exercicio, meta, uid)
        except Exception as erro:
            print(f"[Treino] Erro: {erro}")
            _texto_status(f"Erro no treino: {erro}", "red")

        root.deiconify()
        entry_id.delete(0, tk.END)
        if resumo is not None and resumo.total > 0:
            painel = DashboardTreino(root, resumo)
            painel.mostrar_modal()
            _texto_status(
                f"Sessao finalizada: score {resumo.score:.1f}% "
                f"({resumo.boas}/{resumo.total} boas)",
                "green",
            )
        elif resumo is not None:
            _texto_status("Sessao encerrada sem repeticoes validas.", "orange")
        else:
            _texto_status("Falha ao iniciar o treino. Veja o console.", "red")
        root.lift()
        root.focus_force()
        root.update()
    finally:
        sessao_ativa = False


def processar_uid(uid: str) -> None:
    """Valida o UID no banco e inicia a sessao de treino."""
    uid = (uid or "").strip().upper()
    if not uid:
        return
    if sessao_ativa:
        return
    if conn is None:
        _texto_status("Banco de dados indisponivel.", "red")
        return

    aluno = db.buscar_aluno(conn, uid)
    if aluno is None:
        print(f"[RFID] Cartao nao cadastrado: {uid}")
        _texto_status(f"Cartao nao autorizado: {uid}", "red")
        return

    print(f"[RFID] Acesso liberado: {aluno['nome']} ({uid}) - {aluno['exercicio']}")
    _texto_status(f"Acesso liberado! Bem-vindo, {aluno['nome']}.", "green")
    if root is not None:
        root.update()
    abrir_sessao(uid, aluno["nome"], aluno["exercicio"], int(aluno["repeticoes"]))


def verificar_aluno(event=None) -> None:
    """Fallback: login manual pelo ID digitado."""
    if entry_id is None:
        return
    uid = entry_id.get().strip().upper()
    if not uid:
        _texto_status("Digite um ID ou aproxime o cartao.", "orange")
        return
    processar_uid(uid)


def verificar_arduino() -> None:
    """Polling da porta serial: le o UID enviado pelo Arduino."""
    global buffer_serial

    try:
        if arduino is None or not getattr(arduino, "is_open", False):
            tentar_conectar_serial()
        elif arduino.in_waiting > 0:
            bruto = arduino.read(arduino.in_waiting).decode("utf-8", errors="ignore")
            buffer_serial += bruto
            while "\n" in buffer_serial:
                linha, buffer_serial = buffer_serial.split("\n", 1)
                linha = linha.strip().upper()
                if linha:
                    processar_uid(linha)
    except Exception as erro:
        print(f"[RFID] Erro na leitura serial: {erro}")

    if root is not None:
        root.after(INTERVALO_LEITURA_SERIAL_MS, verificar_arduino)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def atualizar_status_mqtt() -> None:
    if label_mqtt is None:
        return
    if mqtt_cliente is not None and mqtt_cliente.conectado:
        label_mqtt.config(text=f"MQTT: conectado ({MQTT_HOST}:{MQTT_PORT})", fg="green")
    elif mqtt_cliente is not None:
        label_mqtt.config(text=f"MQTT: offline ({MQTT_HOST}:{MQTT_PORT})", fg="orange")
    else:
        label_mqtt.config(text="MQTT: desativado (--sem-mqtt)", fg="gray")
    if root is not None:
        root.after(1000, atualizar_status_mqtt)


def montar_interface() -> tk.Tk:
    global root, label_status, label_mqtt, label_modelo, entry_id

    root = tk.Tk()
    root.title("Smart Gym - Estacao 01 (CP6)")
    root.geometry("480x400")
    root.configure(bg="#0b1220")

    tk.Label(root, text="SMART GYM", bg="#0b1220", fg="#e8eefc",
             font=("Segoe UI", 20, "bold")).pack(pady=(18, 0))
    tk.Label(root, text="IoB - classificacao de execucao em tempo real", bg="#0b1220",
             fg="#8ea0bf", font=("Segoe UI", 9)).pack()

    label_status = tk.Label(root, text="Aproxime seu cartao ...", bg="#0b1220",
                            fg="#e8eefc", font=("Segoe UI", 13))
    label_status.pack(pady=14)

    label_mqtt = tk.Label(root, text="", bg="#0b1220", fg="gray", font=("Segoe UI", 8))
    label_mqtt.pack(pady=2)

    label_modelo = tk.Label(
        root, text="", bg="#0b1220", fg="#38bdf8", font=("Segoe UI", 8), wraplength=430,
    )
    label_modelo.pack(pady=(0, 8))
    if classificador is not None:
        label_modelo.config(text=f"ML: {classificador.descricao_modelo()}")

    tk.Label(root, text="Ou digite o ID do aluno:", bg="#0b1220", fg="#e8eefc",
             font=("Segoe UI", 10)).pack(pady=(6, 2))

    entry_id = tk.Entry(root, font=("Segoe UI", 14), justify="center", width=18,
                        bg="#121c2e", fg="#e8eefc", insertbackground="#e8eefc",
                        relief="flat")
    entry_id.pack(pady=6, ipady=6)

    tk.Button(root, text="Iniciar Exercicio", command=verificar_aluno,
              font=("Segoe UI", 11, "bold"), bg="#38bdf8", fg="#04121f",
              activebackground="#7dd3fc", activeforeground="#04121f",
              relief="flat", padx=20, pady=8, cursor="hand2").pack(pady=10)

    tk.Label(root, text="python index.py --demo  -> demonstracao sem camera",
             bg="#0b1220", fg="#8ea0bf", font=("Segoe UI", 8)).pack(pady=(4, 0))

    root.bind("<Return>", verificar_aluno)
    return root


# ---------------------------------------------------------------------------
# Modo demonstracao (sem camera)
# ---------------------------------------------------------------------------
def gerar_features_demo(rng: random.Random, tipo: str) -> dict[str, float]:
    """Gera uma repeticao plausivel para validar o fluxo sem webcam."""
    if tipo == "boa":
        angulo_min = rng.uniform(44, 56)
        angulo_max = rng.uniform(163, 176)
        duracao = rng.uniform(1.8, 2.5)
        concentrica = duracao * rng.uniform(0.40, 0.52)
        vel_media = rng.uniform(95, 125)
        vel_max = vel_media * rng.uniform(1.5, 1.9)
        vel_std = rng.uniform(8, 17)
        pico = rng.uniform(0.05, 0.20)
        tronco_medio = rng.uniform(2, 7)
        tronco_std = rng.uniform(1.2, 3.2)
        abducao = rng.uniform(14, 24)
        deriva = rng.uniform(0.05, 0.14)
        confianca = rng.uniform(0.86, 0.98)
        intervalo = rng.uniform(2.2, 3.2)
    else:
        angulo_min = rng.uniform(65, 82)
        angulo_max = rng.uniform(148, 163)
        duracao = rng.uniform(2.8, 4.2)
        concentrica = duracao * rng.uniform(0.30, 0.45)
        vel_media = rng.uniform(55, 82)
        vel_max = vel_media * rng.uniform(1.8, 2.4)
        vel_std = rng.uniform(24, 44)
        pico = rng.uniform(0.25, 0.60)
        tronco_medio = rng.uniform(9, 17)
        tronco_std = rng.uniform(6, 11)
        abducao = rng.uniform(26, 40)
        deriva = rng.uniform(0.24, 0.44)
        confianca = rng.uniform(0.78, 0.95)
        intervalo = rng.uniform(3.6, 5.4)

    return {
        "angulo_min": angulo_min,
        "angulo_max": angulo_max,
        "amplitude": angulo_max - angulo_min,
        "duracao_s": duracao,
        "duracao_concentrica_s": concentrica,
        "vel_media": vel_media,
        "vel_max": vel_max,
        "vel_std": vel_std,
        "tempo_no_pico_s": pico,
        "inclinacao_tronco_media": tronco_medio,
        "inclinacao_tronco_std": tronco_std,
        "abducao_ombro_media": abducao,
        "deriva_cotovelo": deriva,
        "confianca_media": confianca,
        "intervalo_anterior_s": intervalo,
    }


def executar_demo(args: argparse.Namespace) -> int:
    if classificador is None or conn is None:
        print("[Demo] Recursos nao inicializados.")
        return 1

    rng = random.Random(args.demo_seed)
    planejado = ["boa"] * args.demo_boas + ["fadiga"] * args.demo_ruins
    if not planejado:
        print("[Demo] Informe ao menos uma repeticao (--demo-boas/--demo-ruins).")
        return 1
    rng.shuffle(planejado)

    aluno, uid, exercicio = "Aluno Demo", "CP6DEMO1", "Rosca Direita"
    meta = len(planejado)
    id_treino = db.criar_treino(conn, uid, exercicio, meta, classificador.descricao_modelo())
    reps: list[RepResumo] = []
    boas = 0
    acertos = 0
    duracao_total = 0.0

    print("=" * 74)
    print(" SmartGym - MODO DEMONSTRACAO (repeticoes simuladas, sem camera)")
    print("=" * 74)
    print(f" Modelo: {classificador.descricao_modelo()}")

    for numero, esperado in enumerate(planejado, start=1):
        features = gerar_features_demo(rng, esperado)
        resultado = classificador.classificar(features)
        duracao_total += features["intervalo_anterior_s"]
        boas += 1 if resultado.classificacao == "BOA" else 0
        ruins = numero - boas
        score = 100.0 * boas / numero
        acertos += 1 if resultado.classificacao == ("BOA" if esperado == "boa" else "RUIM") else 0

        db.registrar_repeticao(
            conn, id_treino, numero, resultado.classificacao,
            resultado.probabilidade, features,
        )
        publicar_repeticao(
            mqtt_cliente,
            aluno=aluno, uid=uid, exercicio=exercicio, meta=meta,
            numero=numero, fase="EXTENDIDO", angulo=features["angulo_min"],
            garrafa_visivel=True, classificacao=resultado.classificacao,
            probabilidade=resultado.probabilidade, boas=boas, ruins=ruins, score=score,
        )
        reps.append(RepResumo(
            numero=numero,
            classificacao=resultado.classificacao,
            probabilidade=resultado.probabilidade,
            angulo_min=features["angulo_min"],
            angulo_max=features["angulo_max"],
        ))
        print(
            f"[Demo] rep {numero:02d}/{meta}: esperado={esperado:6s} "
            f"previsto={resultado.classificacao:4s} ({resultado.percentual:3d}%)"
        )

    total = len(reps)
    boas = sum(1 for rep in reps if rep.classificacao == "BOA")
    ruins = total - boas
    score_final = 100.0 * boas / total if total else 0.0
    db.finalizar_treino(conn, id_treino, boas, ruins, score_final)
    publicar_fim_treino(
        mqtt_cliente,
        aluno=aluno, uid=uid, exercicio=exercicio, meta=meta,
        boas=boas, ruins=ruins, score=score_final, duracao_s=duracao_total,
        modelo=classificador.descricao_modelo(),
    )

    print("-" * 74)
    print(f" Resultado: {boas} BOAS / {ruins} RUINS -> Score {score_final:.1f}%")
    print(f" Acerto da classificacao vs. rotulo esperado: {acertos}/{total}")
    print("=" * 74)

    resumo = ResumoTreino(
        aluno=aluno, uid=uid, exercicio=exercicio, meta=meta, reps=reps,
        modelo=classificador.descricao_modelo(), duracao_s=duracao_total,
        id_treino=id_treino, origem="demo",
    )

    janela = tk.Tk()
    janela.title("Smart Gym - demo")
    janela.geometry("480x150")
    janela.configure(bg="#0b1220")
    tk.Label(janela, text="MODO DEMONSTRACAO", bg="#0b1220", fg="#38bdf8",
             font=("Segoe UI", 14, "bold")).pack(pady=(24, 4))
    tk.Label(janela, text="Feche o dashboard para encerrar.", bg="#0b1220",
             fg="#8ea0bf", font=("Segoe UI", 9)).pack()
    janela.withdraw()  # apenas o dashboard fica visivel

    painel = DashboardTreino(
        janela,
        resumo,
        salvar_em=args.png or None,
        auto_fechar_ms=args.auto_fechar or None,
    )
    painel.mostrar_modal()
    janela.destroy()
    return 0


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Gym - estacao com IoB (CP6).")
    parser.add_argument("--demo", action="store_true",
                        help="Executa uma sessao simulada sem camera.")
    parser.add_argument("--demo-boas", type=int, default=8,
                        help="Repeticoes BOAS simuladas no modo demo (padrao: 8).")
    parser.add_argument("--demo-ruins", type=int, default=2,
                        help="Repeticoes com fadiga simuladas no modo demo (padrao: 2).")
    parser.add_argument("--demo-seed", type=int, default=42, help="Semente do modo demo.")
    parser.add_argument("--sem-mqtt", action="store_true", help="Nao conecta ao broker MQTT.")
    parser.add_argument("--png", default="", help="No modo demo, salva o dashboard neste PNG.")
    parser.add_argument("--auto-fechar", type=int, default=0,
                        help="No modo demo, fecha o dashboard apos N milissegundos.")
    parser.add_argument("--porta", default=PORTA_COM, help=f"Porta serial do Arduino (padrao: {PORTA_COM}).")
    parser.add_argument("--camera", type=int, default=CAMERA, help="Indice da webcam.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    global conn, detector_pose, modelo_yolo, classificador, mqtt_cliente
    global PORTA_COM, CAMERA

    args = _parse_args(argv)
    PORTA_COM = args.porta
    CAMERA = args.camera

    print("=" * 74)
    print(" SmartGym - Estacao 01 | CP6 - Inteligencia de Comportamento")
    print(f" MQTT: {MQTT_HOST}:{MQTT_PORT} | Topico: {MQTT_TOPIC_BASE}/<UID>/exercicio")
    print("=" * 74)

    conn = db.conectar()
    db.inicializar_schema(conn)
    classificador = ClassificadorRepeticoes()

    if not args.sem_mqtt:
        mqtt_cliente = ClienteMQTT()
        mqtt_cliente.conectar()
    else:
        print("[MQTT] Desativado por --sem-mqtt.")

    if args.demo:
        try:
            return executar_demo(args)
        finally:
            if mqtt_cliente is not None:
                mqtt_cliente.desconectar()
            if conn is not None:
                conn.close()

    # ---- modo camera ------------------------------------------------------
    if not os.path.exists(MODELO_POSE):
        print(f"[Erro] Modelo de pose ausente: {MODELO_POSE}")
        print("       Coloque o arquivo na raiz do projeto (veja o README).")
        return 1
    if not os.path.exists(MODELO_YOLO):
        print(f"[Aviso] {MODELO_YOLO} nao encontrado; o Ultralytics tentara baixar automaticamente.")

    base_options = python.BaseOptions(model_asset_path=MODELO_POSE)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO
    )
    detector_pose = vision.PoseLandmarker.create_from_options(options)
    modelo_yolo = YOLO(MODELO_YOLO)

    janela = montar_interface()
    tentar_conectar_serial(forcar=True)
    janela.after(INTERVALO_LEITURA_SERIAL_MS, verificar_arduino)
    janela.after(500, atualizar_status_mqtt)

    try:
        janela.mainloop()
    finally:
        if arduino is not None:
            try:
                arduino.close()
            except Exception:
                pass
        if mqtt_cliente is not None:
            mqtt_cliente.desconectar()
        if conn is not None:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
