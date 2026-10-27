"""Persistencia SQLite do Smart Gym (CP6).

Alem das tabelas originais (`alunos` e `acessos`), este modulo cria:

- `treinos`    -> uma linha por sessao de treino, com o Score de Eficiencia.
- `repeticoes` -> uma linha por repeticao classificada (BOA / RUIM), com as
                  features usadas pelo modelo em JSON.

Tudo e criado de forma idempotente (`CREATE TABLE IF NOT EXISTS`), entao rodar
`init_db.py` em um banco ja existente nao apaga nada.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional

CAMINHO_DB = "smartgym.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS alunos (
    id TEXT,
    nome TEXT,
    exercicio TEXT,
    repeticoes INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_alunos_id ON alunos(id);

CREATE TABLE IF NOT EXISTS acessos (
    id_acesso INTEGER PRIMARY KEY AUTOINCREMENT,
    uid_aluno TEXT,
    horario_acesso TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS treinos (
    id_treino INTEGER PRIMARY KEY AUTOINCREMENT,
    uid_aluno TEXT NOT NULL,
    exercicio TEXT,
    repeticoes_meta INTEGER,
    repeticoes_boas INTEGER DEFAULT 0,
    repeticoes_ruins INTEGER DEFAULT 0,
    score_eficiencia REAL,
    modelo TEXT,
    inicio TIMESTAMP,
    fim TIMESTAMP
);

CREATE TABLE IF NOT EXISTS repeticoes (
    id_rep INTEGER PRIMARY KEY AUTOINCREMENT,
    id_treino INTEGER NOT NULL,
    numero INTEGER,
    classificacao TEXT,
    probabilidade REAL,
    angulo_min REAL,
    angulo_max REAL,
    features_json TEXT,
    timestamp TIMESTAMP,
    FOREIGN KEY (id_treino) REFERENCES treinos (id_treino) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_repeticoes_treino ON repeticoes(id_treino);
CREATE INDEX IF NOT EXISTS idx_treinos_aluno ON treinos(uid_aluno);
"""

ALUNOS_SEED = [
    ("A95D9B99", "Naka", "Rosca Direita", 10),
    ("CP6DEMO1", "Aluno Demo", "Rosca Direita", 10),
]


def agora() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def conectar(caminho: str = CAMINHO_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(caminho)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def inicializar_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def seed_alunos(conn: sqlite3.Connection, alunos=ALUNOS_SEED) -> int:
    """Insere alunos de exemplo sem duplicar (usa o indice unico de `id`)."""
    inseridos = 0
    for uid, nome, exercicio, meta in alunos:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO alunos (id, nome, exercicio, repeticoes) VALUES (?, ?, ?, ?)",
            (uid, nome, exercicio, int(meta)),
        )
        inseridos += cursor.rowcount if cursor.rowcount > 0 else 0
    conn.commit()
    return inseridos


def buscar_aluno(conn: sqlite3.Connection, uid: str) -> Optional[dict]:
    linha = conn.execute(
        "SELECT id, nome, exercicio, repeticoes FROM alunos WHERE id = ?", (uid,)
    ).fetchone()
    return dict(linha) if linha else None


def registrar_acesso(conn: sqlite3.Connection, uid: str) -> None:
    conn.execute("INSERT INTO acessos (uid_aluno) VALUES (?)", (uid,))
    conn.commit()


def criar_treino(conn: sqlite3.Connection, uid: str, exercicio: str, meta: int, modelo: str) -> int:
    cursor = conn.execute(
        """INSERT INTO treinos (uid_aluno, exercicio, repeticoes_meta, modelo, inicio)
           VALUES (?, ?, ?, ?, ?)""",
        (uid, exercicio, int(meta), modelo, agora()),
    )
    conn.commit()
    return int(cursor.lastrowid)


def registrar_repeticao(
    conn: sqlite3.Connection,
    id_treino: int,
    numero: int,
    classificacao: str,
    probabilidade: float,
    features: dict[str, float],
) -> None:
    conn.execute(
        """INSERT INTO repeticoes
           (id_treino, numero, classificacao, probabilidade, angulo_min, angulo_max,
            features_json, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            int(id_treino),
            int(numero),
            classificacao,
            float(probabilidade),
            float(features.get("angulo_min", 0.0)),
            float(features.get("angulo_max", 0.0)),
            json.dumps(features, ensure_ascii=False),
            agora(),
        ),
    )
    conn.commit()


def finalizar_treino(
    conn: sqlite3.Connection, id_treino: int, boas: int, ruins: int, score: float
) -> None:
    conn.execute(
        """UPDATE treinos
              SET repeticoes_boas = ?, repeticoes_ruins = ?, score_eficiencia = ?, fim = ?
            WHERE id_treino = ?""",
        (int(boas), int(ruins), round(float(score), 1), agora(), int(id_treino)),
    )
    conn.commit()


def historico_treinos(conn: sqlite3.Connection, limite: int = 10) -> list[dict]:
    linhas = conn.execute(
        """SELECT t.id_treino, t.uid_aluno, a.nome, t.exercicio, t.repeticoes_meta,
                  t.repeticoes_boas, t.repeticoes_ruins, t.score_eficiencia, t.modelo,
                  t.inicio, t.fim
             FROM treinos t
             LEFT JOIN alunos a ON a.id = t.uid_aluno
            ORDER BY t.id_treino DESC
            LIMIT ?""",
        (int(limite),),
    ).fetchall()
    return [dict(linha) for linha in linhas]


def estatisticas_aluno(conn: sqlite3.Connection, uid: str) -> dict:
    linha = conn.execute(
        """SELECT COUNT(*) AS sessoes,
                  COALESCE(AVG(score_eficiencia), 0) AS score_medio,
                  COALESCE(SUM(repeticoes_boas), 0) AS total_boas,
                  COALESCE(SUM(repeticoes_ruins), 0) AS total_ruins
             FROM treinos
            WHERE uid_aluno = ? AND score_eficiencia IS NOT NULL""",
        (uid,),
    ).fetchone()
    return dict(linha) if linha else {"sessoes": 0, "score_medio": 0, "total_boas": 0, "total_ruins": 0}


def descricao_treino(conn: sqlite3.Connection, id_treino: int) -> Optional[dict]:
    linha = conn.execute("SELECT * FROM treinos WHERE id_treino = ?", (id_treino,)).fetchone()
    if not linha:
        return None
    dados = dict(linha)
    reps = conn.execute(
        """SELECT numero, classificacao, probabilidade, angulo_min, angulo_max, features_json
             FROM repeticoes WHERE id_treino = ? ORDER BY numero""",
        (id_treino,),
    ).fetchall()
    dados["repeticoes"] = [dict(r) for r in reps]
    return dados


if __name__ == "__main__":
    conexao = conectar()
    inicializar_schema(conexao)
    seed_alunos(conexao)
    print("Banco pronto:", CAMINHO_DB)
    for aluno in conexao.execute("SELECT * FROM alunos").fetchall():
        print(" -", dict(aluno))
    conexao.close()
