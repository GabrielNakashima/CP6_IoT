"""Cliente MQTT e contrato de mensagens do Smart Gym (CP6).

Topico por repeticao : smartgym/aluno/<UID>/exercicio
Topico de sessao     : smartgym/aluno/<UID>/treino

Cada repeticao agora leva a classificacao do modelo de Machine Learning
(BOA ou RUIM), a probabilidade associada e o Score de Eficiencia parcial.
Ao final do treino e publicado um evento `fim_treino` com o score final.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Optional

import paho.mqtt.client as mqtt

MQTT_HOST = "3.135.234.101"
MQTT_PORT = 1883
MQTT_USER = ""
MQTT_PASSWORD = ""
MQTT_KEEPALIVE = 60
MQTT_TIMEOUT_CONEXAO = 5
MQTT_TOPIC_BASE = "smartgym/aluno"
MQTT_QOS = 1


class ClienteMQTT:
    """Envelopa o paho-mqtt para publicar os dados do exercicio.

    Nunca levanta excecao por broker fora do ar: o treino continua localmente e
    as mensagens ficam na fila do paho para quando a conexao voltar.
    """

    def __init__(self, host: str = MQTT_HOST, porta: int = MQTT_PORT):
        self.host = host
        self.porta = porta
        client_id = f"smartgym-estacao01-{int(time.time())}"
        try:
            self.cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except AttributeError:  # paho-mqtt < 2.0
            self.cliente = mqtt.Client(client_id=client_id)

        if MQTT_USER:
            self.cliente.username_pw_set(MQTT_USER, MQTT_PASSWORD)

        self.cliente.on_connect = self._on_connect
        self.cliente.on_disconnect = self._on_disconnect
        self.conectado = False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self.conectado = True
            print(f"[MQTT] Conectado em {self.host}:{self.porta}")
        else:
            self.conectado = False
            print(f"[MQTT] Falha na conexao: {reason_code}")

    def _on_disconnect(self, client, userdata, *args):
        self.conectado = False
        print("[MQTT] Desconectado do broker.")

    def conectar(self) -> bool:
        try:
            self.cliente.connect_async(self.host, self.porta, MQTT_KEEPALIVE)
            self.cliente.loop_start()

            limite = time.time() + MQTT_TIMEOUT_CONEXAO
            while time.time() < limite and not self.conectado:
                time.sleep(0.1)

            if self.conectado:
                return True

            print(
                f"[MQTT] Broker {self.host}:{self.porta} ainda nao respondeu. "
                "O treino continua funcionando e os dados serao enviados assim que a conexao subir."
            )
            return False
        except Exception as erro:
            print(f"[MQTT] Erro ao conectar: {erro}")
            return False

    def publicar(self, topico: str, payload: dict) -> bool:
        try:
            info = self.cliente.publish(
                topico, json.dumps(payload, ensure_ascii=False), qos=MQTT_QOS
            )
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] Falha ao publicar em '{topico}' (rc={info.rc}).")
                return False
            return True
        except Exception as erro:
            print(f"[MQTT] Erro ao publicar: {erro}")
            return False

    def desconectar(self) -> None:
        try:
            self.cliente.loop_stop()
            self.cliente.disconnect()
        except Exception:
            pass


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def publicar_repeticao(
    cliente: Optional[ClienteMQTT],
    *,
    aluno: str,
    uid: str,
    exercicio: str,
    meta: int,
    numero: int,
    fase: str,
    angulo: float,
    garrafa_visivel: bool,
    classificacao: str,
    probabilidade: float,
    boas: int,
    ruins: int,
    score: float,
) -> bool:
    """Publica uma repeticao ja classificada."""
    if cliente is None:
        return False
    topico = f"{MQTT_TOPIC_BASE}/{uid}/exercicio"
    payload = {
        "evento": "repeticao",
        "aluno": aluno,
        "uid": uid,
        "exercicio": exercicio,
        "repeticoes_meta": int(meta),
        "repeticoes_feitas": int(numero),
        "fase": fase,
        "angulo_cotovelo": round(float(angulo), 2),
        "garrafa_visivel": bool(garrafa_visivel),
        # --- Novos campos do CP6 (IoB) ---
        "classificacao": classificacao,          # "BOA" ou "RUIM"
        "probabilidade_boa": round(float(probabilidade), 4),
        "repeticoes_boas": int(boas),
        "repeticoes_ruins": int(ruins),
        "score_eficiencia": round(float(score), 1),
        "timestamp": _timestamp(),
    }
    publicado = cliente.publicar(topico, payload)
    if publicado:
        print(
            f"[MQTT] {topico} -> rep {numero}/{meta} | {classificacao} "
            f"({probabilidade * 100:.0f}%) | score {score:.1f}%"
        )
    return publicado


def publicar_fim_treino(
    cliente: Optional[ClienteMQTT],
    *,
    aluno: str,
    uid: str,
    exercicio: str,
    meta: int,
    boas: int,
    ruins: int,
    score: float,
    duracao_s: float,
    modelo: str,
) -> bool:
    """Publica o encerramento da sessao com o Score de Eficiencia final."""
    if cliente is None:
        return False
    topico = f"{MQTT_TOPIC_BASE}/{uid}/treino"
    payload = {
        "evento": "fim_treino",
        "aluno": aluno,
        "uid": uid,
        "exercicio": exercicio,
        "repeticoes_meta": int(meta),
        "repeticoes_boas": int(boas),
        "repeticoes_ruins": int(ruins),
        "total_repeticoes": int(boas + ruins),
        "score_eficiencia": round(float(score), 1),
        "duracao_s": round(float(duracao_s), 2),
        "modelo": modelo,
        "timestamp": _timestamp(),
    }
    publicado = cliente.publicar(topico, payload)
    if publicado:
        print(f"[MQTT] {topico} -> fim_treino | score final {score:.1f}%")
    return publicado
