# Inteligência de Comportamento (IoB) — documentação do Machine Learning

Este documento explica **como o Smart Gym deixa de ser um contador e passa a ser um avaliador** de execução, no formato exigido pelo Checkpoint 6.

> Resumo: cada repetição de rosca direita vira um vetor de 15 características (features). Um modelo de classificação treinado com exemplos reais classifica a repetição como **BOA** ou **RUIM (fadiga)**, em tempo real. O conjunto da sessão gera o **Score de Eficiência** no dashboard.

---

## 1. Pipeline completo

```
Webcam ──> MediaPipe Pose ──> ângulo do cotovelo por quadro
                                    │
                                    ▼
                        RastreadorRosca (features.py)
              fecha a repetição quando o cotovelo volta a esticar
                                    │
                                    ▼
                        extrair_features() → 15 números
                                    │
                                    ▼
             ClassificadorRepeticoes (classifier.py + sklearn)
                     modelo_repeticao.joblib  (ou fallback)
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              SQLite (db.py)    MQTT (mqtt_client)  Dashboard (dashboard.py)
              treinos/repeticoes  classificação       Score de Eficiência
```

Arquivos envolvidos:

| Arquivo | Papel |
|---|---|
| `features.py` | Máquina de estados da repetição + extração das 15 features (sem dependência de câmera). |
| `collect_data.py` | Grava repetições rotuladas em `dados_treino.csv`. |
| `train_model.py` | Treina, valida, salva `modelo_repeticao.joblib`, `metricas_modelo.json` e figuras. |
| `classifier.py` | Carrega o modelo e responde `BOA`/`RUIM` + probabilidade, com fallback heurístico. |
| `dashboard.py` | Mostra o Score de Eficiência e o gráfico por repetição. |
| `db.py` | Persiste treinos e repetições (histórico auditável). |
| `index.py` | Integra tudo em tempo real (RFID, câmera, YOLO, ML, MQTT, painel). |

---

## 2. Coleta de dados

Meta: **10 repetições boas + 10 repetições com fadiga**, gravadas com a mesma pessoa, câmera, distância e iluminação da demonstração.

```bash
python collect_data.py --classe boa    --reps 10
python collect_data.py --classe fadiga --reps 10
```

- O modo `--classe` rotula automaticamente cada repetição concluída.
- O modo `--manual` deixa você decidir no teclado (`B` = boa, `R` = fadiga, `D` = descartar).
- Cada linha do CSV contém `timestamp`, `uid`, `label` e as 15 features.
- **Não é necessário** exibir a garrafa na coleta: a classificação usa somente a pose. No treino normal a garrafa continua obrigatória.

### Como produzir uma repetição "com fadiga" de forma convincente

Faça o movimento incompleto ou descontrolado de propósito:

- não estenda/flexione até o fim (amplitude menor);
- faça a subida mais lenta, travando no meio;
- balance o tronco para "roubar" o movimento;
- deixe o cotovelo sair do lugar (afastar do corpo);
- faça pausas maiores entre repetições.

Esses comportamentos são justamente os que as features capturam.

### Boas práticas

- Grave as duas classes **na mesma sessão e no mesmo enquadramento**.
- Evite repetições em que o corpo sai do quadro (a confiança da pose precisa ser ≥ 0,35).
- Se possível, colete 15 por classe: melhora a validação, que é o ponto mais frágil com 20 amostras.

---

## 3. Features (15) e por que cada uma existe

Todas são calculadas sobre a repetição completa (início esticado → flexão → volta a esticar), com distâncias normalizadas pela largura dos ombros — assim a classificação não depende da distância da câmera.

| # | Feature | O que mede | Comportamento na fadiga |
|---|---|---|---|
| 1 | `angulo_min` | menor ângulo do cotovelo | aumenta (flexão incompleta) |
| 2 | `angulo_max` | maior ângulo (extensão) | pode diminuir |
| 3 | `amplitude` | excursão total do movimento | **diminui** |
| 4 | `duracao_s` | tempo da repetição | aumenta |
| 5 | `duracao_concentrica_s` | tempo até o pico de flexão | aumenta |
| 6 | `vel_media` | velocidade angular média | cai |
| 7 | `vel_max` | pico de velocidade | cai/errático |
| 8 | `vel_std` | irregularidade da velocidade | **aumenta** |
| 9 | `tempo_no_pico_s` | tempo parado na contração | aumenta |
| 10 | `inclinacao_tronco_media` | inclinação média do tronco | aumenta |
| 11 | `inclinacao_tronco_std` | balanço do tronco | **aumenta** |
| 12 | `abducao_ombro_media` | abertura do ombro | aumenta (compensação) |
| 13 | `deriva_cotovelo` | deslocamento do cotovelo relativo ao ombro | **aumenta** |
| 14 | `confianca_media` | qualidade da detecção de pose | cai (pose ruim) |
| 15 | `intervalo_anterior_s` | intervalo desde a repetição anterior | aumenta |

A **ordem e os nomes** ficam gravados dentro do `.joblib`. O `classifier.py` remonta o vetor pela lista salva, então mudar a ordem em `FEATURE_NAMES` não quebra um modelo já treinado.

---

## 4. Modelos e validação

`train_model.py` compara dois candidatos:

1. **Random Forest** (`n_estimators=300`, `max_depth=4`, `class_weight="balanced"`) — robusto a features em escalas diferentes e a outliers, funciona bem com poucas amostras e fornece importância das features.
2. **Regressão Logística** (`class_weight="balanced"`, `max_iter=2000`) — modelo linear simples, bom contraponto interpretável.

Ambos dentro de um `Pipeline` com `StandardScaler`.

Validação com 20 amostras:

- **Leave-One-Out (LOO)**: treina 20 vezes, deixando uma amostra de fora e testando nela. É o mais adequado para dataset minúsculo, mas tem variância alta.
- **StratifiedKFold (5)**: confirma se o resultado é estável entre partições.
- Matriz de confusão, acurácia por modelo e importância das features são gravadas em `metricas_modelo.json` e nos PNGs.

O melhor modelo (por acurácia LOO) é reajustado com **todos** os dados e salvo.

### Como ler as métricas com honestidade

Com 20 amostras, uma acurácia de 100% é possível e **não significa** generalização perfeita. No relatório use as palavras certas:

- LOO é uma estimativa com alta variância;
- o dataset é pequeno e do mesmo voluntário/ambiente;
- para produção, o correto é coletar centenas de repetições de vários alunos.

Essa limitação está registrada em `metricas_modelo.json → observacao` e deve aparecer na apresentação. Reconhecer o limite vale mais do que prometer 99%.

### Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `modelo_repeticao.joblib` | pipeline treinado + lista de features + métricas + data |
| `metricas_modelo.json` | dataset (hash, amostras, por classe), candidatos, LOO, KFold, matriz, importâncias |
| `matriz_confusao.png` | figura da matriz de confusão (para o README) |
| `importancia_features.png` | figura com o peso de cada feature |

---

## 5. Classificação em tempo real

A cada quadro:

1. MediaPipe retorna os landmarks; `amostra_de_landmarks()` escolhe o braço com melhor visibilidade e calcula o ângulo do cotovelo.
2. `RastreadorRosca` guarda os quadros do movimento:
   - começa a repetição quando o ângulo cai abaixo de **60°** estando estendido;
   - **fecha** a repetição quando o ângulo volta acima de **160°**.
3. `extrair_features()` gera o vetor.
4. `classificar()` devolve `BOA`/`RUIM` e a probabilidade (limiar 0,5).
5. O resultado aparece no overlay, vai para o SQLite, para o MQTT e para o dashboard.

Contar a repetição apenas na **volta à extensão** é intencional: garante que as features representem o ciclo inteiro e não um movimento parcial.

Regras herdadas do CP5:

- a garrafa precisa ter sido vista durante a repetição; caso contrário ela é **descartada** (não entra no score);
- repetições com menos de 5 quadros ou amplitude < 20° são tratadas como ruído e ignoradas.

---

## 6. Score de Eficiência

```
Score de Eficiência = (repetições BOAS / repetições válidas) × 100
```

Faixas exibidas no dashboard:

| Score | Faixa |
|---|---|
| ≥ 80% | Excelente |
| 60–79,9% | Atenção |
| < 60% | Risco de fadiga |

O score é gravado em `treinos.score_eficiencia` e publicado no MQTT em `fim_treino`, permitindo histórico e dashboards externos.

---

## 7. Fallback heurístico (sem modelo)

Se `modelo_repeticao.joblib` não existir (por exemplo, antes da coleta), o sistema **não quebra**: ele usa uma regra de bolso com as mesmas features.

| Critério | Peso | Nota 0 quando... |
|---|---|---|
| amplitude | 0,28 | amplitude ≤ 60° |
| profundidade da flexão | 0,24 | ângulo mínimo ≥ 95° |
| regularidade da velocidade | 0,20 | coeficiente de variação ≥ 0,55 |
| balanço do tronco | 0,15 | desvio da inclinação ≥ 9° |
| deriva do cotovelo | 0,09 | deriva ≥ 0,35 |
| cadência | 0,04 | intervalo ≥ 10 s |

O dashboard e o banco mostram explicitamente `Heuristico (modelo ainda nao treinado)` quando esse caminho é usado. Assim a demonstração nunca fica sem resposta e fica claro o que está em jogo.

---

## 8. Limitações e evolução

- Dataset pequeno (20 amostras) → modelo sensível a mudanças de câmera, luz e pessoa.
- Uma única articulação principal (cotovelo do lado mais visível); não modela punho/ombro em 3D.
- A classificação é binária; uma evolução natural é incluir "repetição parcial" e "execução com compensação".
- Próximos passos sugeridos: mais alunos, janelas deslizantes (LSTM/1D-CNN), feedback textual automático ("estenda mais o braço") e comparação com o histórico do próprio aluno.

---

## 9. Privacidade (IoB responsável)

- Nenhum frame de vídeo é salvo.
- O banco guarda apenas números (features), a classificação e a probabilidade.
- O vídeo é processado localmente; somente o resultado numérico vai para o MQTT.
- O aluno é identificado pelo UID do cartão, não por imagem.

---

## 10. Reprodução rápida

```bash
python init_db.py

python collect_data.py --classe boa    --reps 10
python collect_data.py --classe fadiga --reps 10

python train_model.py
python index.py --demo          # valida dashboard/MQTT sem câmera
python index.py                 # treino real
```
