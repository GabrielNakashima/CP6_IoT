"""Dashboard final do treino com o Score de Eficiencia - Smart Gym / CP6.

Mostra, ao final de cada sessao:

- Score de Eficiencia (%) = repeticoes BOAS / total de repeticoes validas;
- quantas repeticoes foram BOAS e quantas foram classificadas como RUIM (fadiga);
- grafico de barras com a probabilidade de execucao BOA em cada repeticao;
- rosca com a proporcao boas x fadiga;
- modelo utilizado (Random Forest/Regressao Logistica ou modo heuristico).

O grafico e embutido no Tkinter via matplotlib (backend TkAgg), sem servidor web
e sem processo extra.
"""

from __future__ import annotations

import os
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# Paleta do painel
BG = "#0b1220"
CARD = "#121c2e"
BORDA = "#233149"
TEXTO = "#e8eefc"
TEXTO_FRACO = "#8ea0bf"
VERDE = "#22c55e"
VERMELHO = "#ef4444"
AMBAR = "#f59e0b"
AZUL = "#38bdf8"

FONTE = "Segoe UI"


@dataclass(slots=True)
class RepResumo:
    numero: int
    classificacao: str          # "BOA" ou "RUIM"
    probabilidade: float        # probabilidade de ser BOA
    angulo_min: float = 0.0
    angulo_max: float = 0.0


@dataclass
class ResumoTreino:
    aluno: str
    uid: str
    exercicio: str
    meta: int
    reps: list[RepResumo] = field(default_factory=list)
    modelo: str = "Heuristico"
    duracao_s: float = 0.0
    id_treino: int | None = None
    origem: str = "camera"      # "camera" ou "demo"

    @property
    def total(self) -> int:
        return len(self.reps)

    @property
    def boas(self) -> int:
        return sum(1 for rep in self.reps if rep.classificacao == "BOA")

    @property
    def ruins(self) -> int:
        return self.total - self.boas

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.boas / self.total

    @property
    def faixa(self) -> tuple[str, str]:
        if self.total == 0:
            return ("Sem repeticoes", TEXTO_FRACO)
        if self.score >= 80:
            return ("Excelente", VERDE)
        if self.score >= 60:
            return ("Atencao", AMBAR)
        return ("Risco de fadiga", VERMELHO)


def formatar_duracao(segundos: float) -> str:
    segundos = max(0, int(round(segundos)))
    if segundos < 60:
        return f"{segundos}s"
    return f"{segundos // 60}min {segundos % 60:02d}s"


def _slug(texto: str) -> str:
    limpo = "".join(c if c.isalnum() else "_" for c in texto.strip().lower())
    return "_".join(filter(None, limpo.split("_"))) or "treino"


class DashboardTreino(tk.Toplevel):
    """Janela de resultado. Use `mostrar_modal()` para bloquear ate fechar."""

    def __init__(
        self,
        master: tk.Misc,
        resumo: ResumoTreino,
        salvar_em: str | None = None,
        auto_fechar_ms: int | None = None,
    ) -> None:
        super().__init__(master)
        self.resumo = resumo
        self.fig: Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None

        self.title("Smart Gym - Resultado do treino")
        self.configure(bg=BG)
        self.geometry("1040x660")
        self.minsize(940, 620)
        self.protocol("WM_DELETE_WINDOW", self._fechar)

        self._montar()
        self._desenhar_graficos()
        self.update_idletasks()

        if salvar_em:
            self.salvar_png(salvar_em, silencioso=True)
        if auto_fechar_ms:
            self.after(int(auto_fechar_ms), self._fechar)

    # -- construcao da UI --------------------------------------------------
    def _montar(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        self._montar_cabecalho()
        self._montar_cartao_score()
        self._montar_area_grafico()
        self._montar_rodape()

    def _montar_cabecalho(self) -> None:
        cabecalho = tk.Frame(self, bg=BG)
        cabecalho.grid(row=0, column=0, columnspan=2, sticky="ew", padx=22, pady=(18, 8))
        cabecalho.columnconfigure(0, weight=1)

        tk.Label(
            cabecalho, text="RESULTADO DO TREINO", bg=BG, fg=TEXTO_FRACO,
            font=(FONTE, 9, "bold"),
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            cabecalho,
            text=f"{self.resumo.aluno}  |  {self.resumo.exercicio}",
            bg=BG, fg=TEXTO, font=(FONTE, 20, "bold"),
        ).grid(row=1, column=0, sticky="w")

        subtitulo = f"Meta: {self.resumo.meta} repeticoes  -  Duracao: {formatar_duracao(self.resumo.duracao_s)}"
        if self.resumo.origem == "demo":
            subtitulo += "  -  MODO DEMONSTRACAO (dados simulados)"
        tk.Label(
            cabecalho, text=subtitulo, bg=BG, fg=TEXTO_FRACO, font=(FONTE, 10),
        ).grid(row=2, column=0, sticky="w", pady=(2, 0))

        badge = tk.Label(
            cabecalho,
            text="IoB - Inteligencia de Comportamento",
            bg="#0f2a3a", fg=AZUL, font=(FONTE, 9, "bold"), padx=10, pady=5,
        )
        badge.grid(row=0, column=1, rowspan=3, sticky="e")

    def _montar_cartao_score(self) -> None:
        cartao = tk.Frame(self, bg=CARD, highlightbackground=BORDA, highlightthickness=1)
        cartao.grid(row=1, column=0, sticky="nsw", padx=(22, 12), pady=6, ipadx=22, ipady=16)
        cartao.columnconfigure(0, weight=1)

        tk.Label(
            cartao, text="SCORE DE EFICIENCIA", bg=CARD, fg=TEXTO_FRACO,
            font=(FONTE, 9, "bold"),
        ).grid(row=0, column=0, sticky="w")

        nome_faixa, cor_faixa = self.resumo.faixa
        self.label_score = tk.Label(
            cartao, text=f"{self.resumo.score:.1f}%", bg=CARD, fg=cor_faixa,
            font=(FONTE, 44, "bold"),
        )
        self.label_score.grid(row=1, column=0, sticky="w", pady=(4, 0))

        tk.Label(
            cartao, text=nome_faixa, bg=CARD, fg=cor_faixa, font=(FONTE, 12, "bold"),
        ).grid(row=2, column=0, sticky="w")

        tk.Label(
            cartao,
            text="repeticoes BOAS / total de repeticoes validas",
            bg=CARD, fg=TEXTO_FRACO, font=(FONTE, 8),
        ).grid(row=3, column=0, sticky="w", pady=(2, 14))

        self._linha_metrica(cartao, 4, "Repeticoes boas", str(self.resumo.boas),
                            f"{self.resumo.boas / self.resumo.total * 100:.0f}% do treino"
                            if self.resumo.total else "-", VERDE)
        self._linha_metrica(cartao, 5, "Com fadiga (RUIM)", str(self.resumo.ruins),
                            f"{self.resumo.ruins / self.resumo.total * 100:.0f}% do treino"
                            if self.resumo.total else "-", VERMELHO)

        separador = tk.Frame(cartao, bg=BORDA, height=1)
        separador.grid(row=6, column=0, sticky="ew", pady=14)

        info = [
            ("Repeticoes validas", f"{self.resumo.total} de {self.resumo.meta}"),
            ("Modelo", self.resumo.modelo),
            ("ID do treino", str(self.resumo.id_treino) if self.resumo.id_treino else "-"),
        ]
        for deslocamento, (rotulo, valor) in enumerate(info):
            self._linha_info(cartao, 7 + deslocamento, rotulo, valor)

    def _linha_metrica(self, pai: tk.Misc, linha: int, titulo: str, valor: str,
                       detalhe: str, cor: str) -> None:
        quadro = tk.Frame(pai, bg=CARD)
        quadro.grid(row=linha, column=0, sticky="ew", pady=4)
        quadro.columnconfigure(1, weight=1)

        tk.Label(quadro, text=titulo, bg=CARD, fg=TEXTO, font=(FONTE, 10)).grid(
            row=0, column=0, sticky="w")
        tk.Label(quadro, text=valor, bg=CARD, fg=cor, font=(FONTE, 14, "bold")).grid(
            row=0, column=1, sticky="e", padx=(18, 0))
        tk.Label(quadro, text=detalhe, bg=CARD, fg=TEXTO_FRACO, font=(FONTE, 8)).grid(
            row=1, column=0, columnspan=2, sticky="w")

    def _linha_info(self, pai: tk.Misc, linha: int, rotulo: str, valor: str) -> None:
        quadro = tk.Frame(pai, bg=CARD)
        quadro.grid(row=linha, column=0, sticky="ew", pady=2)
        quadro.columnconfigure(1, weight=1)
        tk.Label(quadro, text=rotulo, bg=CARD, fg=TEXTO_FRACO, font=(FONTE, 9)).grid(
            row=0, column=0, sticky="w")
        tk.Label(quadro, text=valor, bg=CARD, fg=TEXTO, font=(FONTE, 9, "bold"),
                 wraplength=210, justify="right").grid(row=0, column=1, sticky="e")

    def _montar_area_grafico(self) -> None:
        self.area_grafico = tk.Frame(self, bg=CARD, highlightbackground=BORDA,
                                     highlightthickness=1)
        self.area_grafico.grid(row=1, column=1, sticky="nsew", padx=(0, 22), pady=6)
        self.area_grafico.rowconfigure(0, weight=1)
        self.area_grafico.columnconfigure(0, weight=1)

        tk.Label(
            self.area_grafico, text="CLASSIFICACAO REPETICAO A REPETICAO",
            bg=CARD, fg=TEXTO_FRACO, font=(FONTE, 9, "bold"),
        ).grid(row=0, column=0, sticky="nw", padx=14, pady=(12, 0))

        self.quadro_figura = tk.Frame(self.area_grafico, bg=CARD)
        self.quadro_figura.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.quadro_figura.rowconfigure(0, weight=1)
        self.quadro_figura.columnconfigure(0, weight=1)

    def _montar_rodape(self) -> None:
        rodape = tk.Frame(self, bg=BG)
        rodape.grid(row=2, column=0, columnspan=2, sticky="ew", padx=22, pady=(6, 18))
        rodape.columnconfigure(0, weight=1)

        tk.Label(
            rodape,
            text="Score de Eficiencia = repeticoes BOAS / total de repeticoes x 100",
            bg=BG, fg=TEXTO_FRACO, font=(FONTE, 8),
        ).grid(row=0, column=0, sticky="w")

        tk.Button(
            rodape, text="Salvar grafico (PNG)", command=self.salvar_png,
            bg="#1b2a44", fg=TEXTO, activebackground="#24385a", activeforeground=TEXTO,
            relief="flat", font=(FONTE, 9, "bold"), padx=14, pady=7, cursor="hand2",
        ).grid(row=0, column=1, padx=(0, 8))

        tk.Button(
            rodape, text="Fechar", command=self._fechar,
            bg=AZUL, fg="#04121f", activebackground="#7dd3fc", activeforeground="#04121f",
            relief="flat", font=(FONTE, 9, "bold"), padx=22, pady=7, cursor="hand2",
        ).grid(row=0, column=2)

    # -- graficos ----------------------------------------------------------
    def _desenhar_graficos(self) -> None:
        self.fig = Figure(figsize=(6.6, 4.4), dpi=100, facecolor=CARD)
        grade = self.fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.32)
        eixo_barras = self.fig.add_subplot(grade[0, 0])
        eixo_rosca = self.fig.add_subplot(grade[0, 1])

        self._estilizar_eixo(eixo_barras)
        self._desenhar_barras(eixo_barras)
        self._desenhar_rosca(eixo_rosca)

        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.16)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.quadro_figura)
        self.canvas.draw()
        self.canvas.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _estilizar_eixo(self, eixo) -> None:
        eixo.set_facecolor(CARD)
        for lado in eixo.spines.values():
            lado.set_color(BORDA)
        eixo.tick_params(colors=TEXTO_FRACO, labelsize=8)
        eixo.grid(axis="y", color=BORDA, alpha=0.55, linewidth=0.6)
        eixo.set_axisbelow(True)

    def _desenhar_barras(self, eixo) -> None:
        reps = self.resumo.reps
        eixo.set_title("Probabilidade de execucao BOA", color=TEXTO, fontsize=10,
                       fontweight="bold", loc="left", pad=10)

        if not reps:
            eixo.text(0.5, 0.5, "Nenhuma repeticao valida", ha="center", va="center",
                      color=TEXTO_FRACO, transform=eixo.transAxes)
            eixo.set_yticks([])
            return

        numeros = [rep.numero for rep in reps]
        valores = [max(0.02, rep.probabilidade) for rep in reps]
        cores = [VERDE if rep.classificacao == "BOA" else VERMELHO for rep in reps]

        barras = eixo.bar(numeros, valores, color=cores, width=0.62, zorder=3)
        eixo.axhline(0.5, color=TEXTO_FRACO, linestyle="--", linewidth=0.9, alpha=0.8, zorder=2)
        eixo.text(numeros[-1] + 0.55, 0.52, "limiar 50%", color=TEXTO_FRACO,
                  fontsize=7, ha="right", va="bottom")

        for barra, rep in zip(barras, reps):
            eixo.text(
                barra.get_x() + barra.get_width() / 2,
                min(0.97, rep.probabilidade + 0.05),
                f"{rep.probabilidade * 100:.0f}%",
                ha="center", va="bottom", color=TEXTO, fontsize=7,
            )

        eixo.set_ylim(0, 1.08)
        eixo.set_xlim(0.3, max(numeros) + 0.9)
        eixo.set_ylabel("confianca de execucao BOA", color=TEXTO_FRACO, fontsize=8)
        eixo.set_xlabel("repeticao", color=TEXTO_FRACO, fontsize=8)
        eixo.set_xticks(numeros)
        if len(numeros) > 12:
            eixo.set_xticks(numeros[::2])
        eixo.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        eixo.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

        eixo.legend(
            handles=[Patch(color=VERDE, label="BOA"), Patch(color=VERMELHO, label="RUIM (fadiga)")],
            loc="upper right", frameon=False, fontsize=8, labelcolor=TEXTO, ncol=2,
        )

    def _desenhar_rosca(self, eixo) -> None:
        eixo.set_facecolor(CARD)
        eixo.set_title("Boas x fadiga", color=TEXTO, fontsize=10,
                       fontweight="bold", loc="left", pad=10)
        eixo.set_aspect("equal")

        boas = self.resumo.boas
        ruins = self.resumo.ruins
        nome_faixa, cor_faixa = self.resumo.faixa

        if boas + ruins == 0:
            eixo.text(0.5, 0.5, "-", ha="center", va="center", color=TEXTO_FRACO,
                      transform=eixo.transAxes, fontsize=20)
            eixo.set_axis_off()
            return

        valores = [boas, ruins]
        eixo.pie(
            valores,
            colors=[VERDE, VERMELHO],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.34, "edgecolor": CARD, "linewidth": 2},
        )
        eixo.text(0, 0.08, f"{self.resumo.score:.0f}%", ha="center", va="center",
                  color=cor_faixa, fontsize=22, fontweight="bold")
        eixo.text(0, -0.18, "BOAS", ha="center", va="center", color=TEXTO_FRACO, fontsize=7)
        eixo.text(0.02, -0.62, f"{boas} boas  -  {ruins} fadiga", ha="center", va="center",
                  color=TEXTO_FRACO, fontsize=8)
        eixo.text(0.02, -0.46, nome_faixa, ha="center", va="center", color=cor_faixa,
                  fontsize=8, fontweight="bold")
        eixo.set_xticks([])
        eixo.set_yticks([])
        for lado in eixo.spines.values():
            lado.set_visible(False)

    # -- acoes -------------------------------------------------------------
    def salvar_png(self, caminho: str | None = None, silencioso: bool = False) -> str | None:
        if self.fig is None:
            return None
        if caminho is None:
            nome_padrao = (
                f"resultado_treino_{_slug(self.resumo.aluno)}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            caminho = filedialog.asksaveasfilename(
                parent=self,
                title="Salvar grafico do treino",
                defaultextension=".png",
                initialfile=nome_padrao,
                filetypes=[("Imagem PNG", "*.png")],
            )
            if not caminho:
                return None

        try:
            self.fig.savefig(caminho, dpi=170, facecolor=CARD)
            if not silencioso:
                messagebox.showinfo(
                    "Grafico salvo", f"Imagem salva em:\n{os.path.abspath(caminho)}",
                    parent=self,
                )
            print(f"[Dashboard] Grafico salvo em {caminho}")
            return caminho
        except Exception as erro:
            if not silencioso:
                messagebox.showerror("Erro ao salvar", str(erro), parent=self)
            else:
                print(f"[Dashboard] Falha ao salvar grafico: {erro}")
            return None

    def _fechar(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()

    def mostrar_modal(self) -> None:
        try:
            self.transient(self.master)
            self.grab_set()
            self.focus_force()
            self.wait_window()
        except tk.TclError:
            pass
