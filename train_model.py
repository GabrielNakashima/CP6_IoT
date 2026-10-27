"""Treinamento e validacao do modelo de classificacao de repeticao (CP6).

Uso:

    python train_model.py

Entradas : dados_treino.csv (gerado por collect_data.py)
Saidas   : modelo_repeticao.joblib    (usado por index.py)
           metricas_modelo.json       (metricas da validacao)
           matriz_confusao.png        (figura para o README)
           importancia_features.png   (figura para o README)

Estrategia para um dataset pequeno (10 boas + 10 com fadiga):
- comparar Random Forest e Regressao Logistica;
- validar com Leave-One-Out (LOO), que treina N vezes deixando uma amostra de
  fora, e tambem com StratifiedKFold(5);
- escolher o melhor LOO, reajustar com todos os dados e salvar o pipeline
  completo (normalizacao + modelo), junto com a lista de features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import DESCRICOES_FEATURES, FEATURE_NAMES

ROTULOS = ["boa", "fadiga"]
ALIASES = {
    "boa": "boa", "boas": "boa", "good": "boa", "ok": "boa", "correta": "boa",
    "fadiga": "fadiga", "ruim": "fadiga", "bad": "fadiga", "errada": "fadiga",
}

CAMINHO_CSV = "dados_treino.csv"
CAMINHO_MODELO = "modelo_repeticao.joblib"
CAMINHO_METRICAS = "metricas_modelo.json"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def carregar_dataset(caminho: str) -> tuple[np.ndarray, np.ndarray, Counter, int]:
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Dataset '{caminho}' nao encontrado. Rode antes:\n"
            "  python collect_data.py --classe boa --reps 10\n"
            "  python collect_data.py --classe fadiga --reps 10"
        )

    linhas_x: list[list[float]] = []
    rotulos: list[str] = []
    ignoradas = 0

    with open(caminho, newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        colunas = leitor.fieldnames or []
        faltando = [nome for nome in FEATURE_NAMES if nome not in colunas]
        if faltando:
            raise ValueError(
                f"CSV sem as colunas: {', '.join(faltando)}. "
                "Recolete os dados com a versao atual do collect_data.py."
            )

        for numero, linha in enumerate(leitor, start=2):
            rotulo_bruto = (linha.get("label") or "").strip().lower()
            rotulo = ALIASES.get(rotulo_bruto)
            if rotulo is None:
                ignoradas += 1
                continue
            try:
                valores = [float(linha[nome]) for nome in FEATURE_NAMES]
            except (TypeError, ValueError):
                print(f"[Treino] Linha {numero} ignorada (valor nao numerico).")
                ignoradas += 1
                continue
            if not all(math.isfinite(valor) for valor in valores):
                ignoradas += 1
                continue
            linhas_x.append(valores)
            rotulos.append(rotulo)

    if not linhas_x:
        raise ValueError("Nenhuma amostra valida encontrada no CSV.")

    contagem = Counter(rotulos)
    return np.array(linhas_x, dtype=float), np.array(rotulos), contagem, ignoradas


def validar_dataset(contagem: Counter) -> None:
    print(f"[Treino] Amostras por classe: boa={contagem.get('boa', 0)} | "
          f"fadiga={contagem.get('fadiga', 0)}")
    faltando = [rotulo for rotulo in ROTULOS if contagem.get(rotulo, 0) < 2]
    if faltando:
        raise ValueError(
            f"Cada classe precisa de pelo menos 2 amostras (faltando: {', '.join(faltando)})."
        )
    if min(contagem.get(rotulo, 0) for rotulo in ROTULOS) < 10:
        print("[Treino] AVISO: ideal e ter 10+ repeticoes por classe. "
              "O modelo funciona, mas a avaliacao fica mais fragil.")


# ---------------------------------------------------------------------------
# Modelos e validacao
# ---------------------------------------------------------------------------
def construir_candidatos() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline([
            ("normalizacao", StandardScaler()),
            ("modelo", RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("normalizacao", StandardScaler()),
            ("modelo", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }


def avaliar_candidatos(
    X: np.ndarray, y: np.ndarray, candidatos: dict[str, Pipeline]
) -> tuple[dict, dict, dict]:
    loo = LeaveOneOut()
    minimo_por_classe = min(Counter(y).values())
    n_splits = max(2, min(5, int(minimo_por_classe)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    resultados: dict[str, dict] = {}
    predicoes_loo: dict[str, np.ndarray] = {}

    for nome, modelo in candidatos.items():
        predicoes = cross_val_predict(modelo, X, y, cv=loo)
        acuracia_loo = float(accuracy_score(y, predicoes))
        matriz = confusion_matrix(y, predicoes, labels=ROTULOS)

        try:
            notas_skf = cross_val_score(modelo, X, y, cv=skf, scoring="accuracy")
            skf_media = float(np.mean(notas_skf))
            skf_desvio = float(np.std(notas_skf))
        except Exception as erro:  # folds degenerados com dataset muito pequeno
            print(f"[Treino] StratifiedKFold indisponivel para {nome}: {erro}")
            skf_media = float("nan")
            skf_desvio = float("nan")

        resultados[nome] = {
            "loo_accuracy": round(acuracia_loo, 4),
            "skf_mean": None if math.isnan(skf_media) else round(skf_media, 4),
            "skf_std": None if math.isnan(skf_desvio) else round(skf_desvio, 4),
            "skf_splits": n_splits,
            "matriz_confusao": matriz.tolist(),
        }
        predicoes_loo[nome] = predicoes

    return resultados, predicoes_loo, {"loo": loo, "n_splits": n_splits}


def imprimir_comparativo(resultados: dict) -> None:
    print("\n" + "=" * 74)
    print(" Comparativo de modelos (validacao cruzada)")
    print("=" * 74)
    print(f"{'modelo':<22}{'LOO':>10}{'KFold (media)':>18}{'KFold (desvio)':>17}")
    print("-" * 74)
    for nome, linha in resultados.items():
        skf_media = "-" if linha["skf_mean"] is None else f"{linha['skf_mean'] * 100:.1f}%"
        skf_desvio = "-" if linha["skf_std"] is None else f"{linha['skf_std'] * 100:.1f}%"
        print(f"{nome:<22}{linha['loo_accuracy'] * 100:>9.1f}%{skf_media:>18}{skf_desvio:>17}")
    print("=" * 74)
    print("Matriz de confusao (linhas = real, colunas = previsto):")
    for nome, linha in resultados.items():
        matriz = np.array(linha["matriz_confusao"])
        print(f"\n  {nome}")
        print(f"    real BOA    -> BOA {matriz[0][0]:>3} | FADIGA {matriz[0][1]:>3}")
        print(f"    real FADIGA -> BOA {matriz[1][0]:>3} | FADIGA {matriz[1][1]:>3}")


def escolher_melhor(resultados: dict) -> str:
    def chave(item):
        nome, linha = item
        skf = linha["skf_mean"] if linha["skf_mean"] is not None else 0.0
        return (linha["loo_accuracy"], skf, -len(nome))

    return max(resultados.items(), key=chave)[0]


def importancias(modelo: Pipeline) -> Optional[list[float]]:
    estimador = modelo.named_steps["modelo"]
    if hasattr(estimador, "feature_importances_"):
        return [float(valor) for valor in estimador.feature_importances_]
    if hasattr(estimador, "coef_"):
        return [float(valor) for valor in np.abs(estimador.coef_[0])]
    return None


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def salvar_matriz_confusao(caminho: str, matriz: np.ndarray, titulo: str) -> None:
    figura, eixo = plt.subplots(figsize=(4.8, 4.0), dpi=150)
    eixo.imshow(matriz, cmap="Blues")

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            cor = "white" if matriz[i, j] > matriz.max() / 2 else "#0f172a"
            eixo.text(j, i, str(int(matriz[i, j])), ha="center", va="center",
                      color=cor, fontsize=16, fontweight="bold")

    eixo.set_xticks([0, 1], ["BOA", "FADIGA"])
    eixo.set_yticks([0, 1], ["BOA", "FADIGA"])
    eixo.set_xlabel("Previsto pelo modelo", fontsize=10)
    eixo.set_ylabel("Rotulo real", fontsize=10)
    eixo.set_title(titulo, fontsize=11, fontweight="bold")
    for lado in eixo.spines.values():
        lado.set_visible(False)
    figura.tight_layout()
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)


def salvar_importancias(caminho: str, nomes: list[str], valores: list[float],
                        titulo: str) -> None:
    ordem = np.argsort(valores)
    figura, eixo = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    eixo.barh([nomes[i] for i in ordem], [valores[i] for i in ordem], color="#2563eb")
    eixo.set_xlabel("Peso na decisao do modelo", fontsize=10)
    eixo.set_title(titulo, fontsize=11, fontweight="bold")
    eixo.grid(axis="x", alpha=0.25)
    eixo.set_axisbelow(True)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    figura.tight_layout()
    figura.savefig(caminho, facecolor="white")
    plt.close(figura)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def treinar(args: argparse.Namespace) -> int:
    print("=" * 74)
    print(" SmartGym - treinamento do classificador de repeticao (CP6)")
    print("=" * 74)

    X, y, contagem, ignoradas = carregar_dataset(args.csv)
    validar_dataset(contagem)
    if ignoradas:
        print(f"[Treino] {ignoradas} linha(s) ignorada(s) por rotulo/valor invalido.")

    candidatos = construir_candidatos()
    resultados, predicoes_loo, _info = avaliar_candidatos(X, y, candidatos)
    imprimir_comparativo(resultados)

    escolhido = escolher_melhor(resultados)
    print(f"\n[Treino] Modelo escolhido: {escolhido} "
          f"(LOO {resultados[escolhido]['loo_accuracy'] * 100:.1f}%)")

    modelo_final = candidatos[escolhido]
    modelo_final.fit(X, y)

    pesos = importancias(modelo_final)
    with open(args.csv, "rb") as arquivo_csv:
        hash_dataset = hashlib.sha256(arquivo_csv.read()).hexdigest()[:16]

    metricas = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "arquivo": os.path.abspath(args.csv),
            "amostras": int(len(y)),
            "por_classe": dict(contagem),
            "sha256": hash_dataset,
        },
        "modelo_escolhido": escolhido,
        "candidatos": resultados,
        "loo_accuracy": resultados[escolhido]["loo_accuracy"],
        "skf_mean": resultados[escolhido]["skf_mean"],
        "skf_std": resultados[escolhido]["skf_std"],
        "matriz_confusao": {
            "labels": ROTULOS,
            "matriz": resultados[escolhido]["matriz_confusao"],
        },
        "n_amostras": int(len(y)),
        "features": FEATURE_NAMES,
        "importancias": (
            {nome: round(valor, 4) for nome, valor in zip(FEATURE_NAMES, pesos)}
            if pesos else None
        ),
        "observacao": (
            "Dataset pequeno (10+10). A acuracia LOO e uma estimativa otimista-pessimista "
            "com alta variancia; use como referencia, nao como garantia."
        ),
    }

    pacote = {
        "modelo": modelo_final,
        "features": FEATURE_NAMES,
        "classes": ROTULOS,
        "modelo_escolhido": escolhido,
        "metricas": metricas,
        "treinado_em": metricas["gerado_em"],
        "versao": 1,
        "limiar_decisao": 0.5,
    }

    joblib.dump(pacote, args.modelo)
    print(f"[Treino] Modelo salvo em {args.modelo}")

    with open(args.metricas, "w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)
    print(f"[Treino] Metricas salvas em {args.metricas}")

    if not args.sem_figuras:
        prefixo = f"{args.prefixo_figuras}_" if args.prefixo_figuras else ""
        matriz = np.array(resultados[escolhido]["matriz_confusao"])
        caminho_matriz = f"{prefixo}matriz_confusao.png"
        salvar_matriz_confusao(
            caminho_matriz, matriz,
            f"Matriz de confusao - {escolhido} (LOO)",
        )
        print(f"[Treino] Figura salva em {caminho_matriz}")

        if pesos:
            caminho_importancias = f"{prefixo}importancia_features.png"
            salvar_importancias(
                caminho_importancias, FEATURE_NAMES, pesos,
                "Features mais relevantes para separar BOA x FADIGA",
            )
            print(f"[Treino] Figura salva em {caminho_importancias}")

    print("\n" + "-" * 74)
    print(" Proximos passos")
    print("-" * 74)
    print(" 1) python index.py --demo      -> valida dashboard/MQTT sem camera")
    print(" 2) python index.py             -> treino real com webcam + RFID")
    print(" 3) preencha as metricas no README.md (secao Machine Learning)")
    print("-" * 74)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Treina o classificador BOA x FADIGA.")
    parser.add_argument("--csv", default=CAMINHO_CSV, help="CSV com as repeticoes rotuladas.")
    parser.add_argument("--modelo", default=CAMINHO_MODELO, help="Saida do modelo (.joblib).")
    parser.add_argument("--metricas", default=CAMINHO_METRICAS, help="Saida das metricas (.json).")
    parser.add_argument("--prefixo-figuras", default="", help="Prefixo dos PNGs gerados.")
    parser.add_argument("--sem-figuras", action="store_true", help="Nao gerar os PNGs.")
    args = parser.parse_args(argv)

    if args.csv == args.modelo:
        print("CSV e modelo nao podem ser o mesmo arquivo.")
        return 2

    try:
        return treinar(args)
    except (FileNotFoundError, ValueError) as erro:
        print(f"\n[Treino] ERRO: {erro}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
