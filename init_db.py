"""Inicializacao do banco SQLite do Smart Gym (CP6).

Cria todas as tabelas necessarias (alunos, acessos, treinos, repeticoes) e
insere os alunos de exemplo. Pode ser executado quantas vezes quiser: nada e
apagado e os alunos existentes nao sao duplicados.

    python init_db.py
"""

from __future__ import annotations

import argparse
import os
import sys

import db

ALUNOS_EXTRA = [
    # (uid, nome, exercicio, meta)
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cria/atualiza o banco do Smart Gym.")
    parser.add_argument("--db", default=db.CAMINHO_DB, help="Caminho do arquivo SQLite.")
    parser.add_argument("--sem-seed", action="store_true", help="Nao inserir alunos de exemplo.")
    args = parser.parse_args(argv)

    existe = os.path.exists(args.db)
    conexao = db.conectar(args.db)
    try:
        db.inicializar_schema(conexao)
        inseridos = 0
        if not args.sem_seed:
            inseridos = db.seed_alunos(conexao, db.ALUNOS_SEED + ALUNOS_EXTRA)

        print("=" * 64)
        print(" SmartGym - banco de dados")
        print("=" * 64)
        print(f" Arquivo : {os.path.abspath(args.db)}")
        print(f" Status  : {'atualizado' if existe else 'criado'}")
        print(f" Alunos novos inseridos: {inseridos}")
        print("-" * 64)

        tabelas = conexao.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        print(" Tabelas : " + ", ".join(linha["name"] for linha in tabelas))

        print(" Alunos  :")
        for aluno in conexao.execute("SELECT id, nome, exercicio, repeticoes FROM alunos ORDER BY nome"):
            print(f"   - {aluno['id']:<10} {aluno['nome']:<16} {aluno['exercicio']:<16} meta {aluno['repeticoes']}")

        total_treinos = conexao.execute("SELECT COUNT(*) AS total FROM treinos").fetchone()["total"]
        total_reps = conexao.execute("SELECT COUNT(*) AS total FROM repeticoes").fetchone()["total"]
        print("-" * 64)
        print(f" Treinos registrados: {total_treinos} | Repeticoes classificadas: {total_reps}")
        print("=" * 64)
        print(" Proximo passo: python collect_data.py --classe boa --reps 10")
        return 0
    finally:
        conexao.close()


if __name__ == "__main__":
    sys.exit(main())
