from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")

RAW_DIR = Path("data/raw")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/learning_examples.md")

EXAMPLES = [
    ("ba48188d", "Caso A: el TVT baja despues del corte"),
    ("a959858c", "Caso B: el TVT sube despues del corte"),
    ("283269ac", "Caso C: el TVT queda casi plano"),
]


def plot_examples() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(14, 10), sharex=False)

    summary_rows = []
    for row_index, (well_id, title) in enumerate(EXAMPLES):
        horizontal = pd.read_csv(RAW_DIR / "train" / f"{well_id}__horizontal_well.csv")
        typewell = pd.read_csv(RAW_DIR / "train" / f"{well_id}__typewell.csv")
        ps_row = int(horizontal["TVT_input"].isna().idxmax())
        hidden = horizontal["TVT_input"].isna()
        hidden_tvt = horizontal.loc[hidden, "TVT"]

        summary_rows.append(
            {
                "well_id": well_id,
                "case": title,
                "total_rows": len(horizontal),
                "prediction_start_row": ps_row,
                "hidden_rows": int(hidden.sum()),
                "hidden_tvt_start": round(float(hidden_tvt.iloc[0]), 2),
                "hidden_tvt_end": round(float(hidden_tvt.iloc[-1]), 2),
                "hidden_tvt_change": round(float(hidden_tvt.iloc[-1] - hidden_tvt.iloc[0]), 2),
            }
        )

        ax_tvt = axes[row_index, 0]
        ax_tvt.plot(horizontal["MD"], horizontal["TVT"], color="#222222", linewidth=1.6, label="TVT real")
        ax_tvt.plot(
            horizontal["MD"],
            horizontal["TVT_input"],
            color="#1f77b4",
            linewidth=2.0,
            label="TVT_input conocido",
        )
        ax_tvt.axvline(horizontal.loc[ps_row, "MD"], color="#d62728", linestyle="--", linewidth=1.5, label="corte")
        ax_tvt.set_title(title)
        ax_tvt.set_xlabel("MD: posicion a lo largo del pozo")
        ax_tvt.set_ylabel("TVT")
        ax_tvt.legend(loc="best", fontsize=8)

        ax_gr = axes[row_index, 1]
        ax_gr.plot(typewell["TVT"], typewell["GR"], color="#444444", linewidth=1.0, label="GR referencia")
        ax_gr.scatter(horizontal["TVT"], horizontal["GR"], s=4, alpha=0.35, color="#2ca02c", label="GR horizontal")
        ax_gr.axvline(horizontal.loc[ps_row, "TVT"], color="#d62728", linestyle="--", linewidth=1.5, label="TVT en corte")
        ax_gr.set_title("Pista GR: patron medido contra TVT")
        ax_gr.set_xlabel("TVT")
        ax_gr.set_ylabel("GR")
        ax_gr.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_examples.png", dpi=160)
    plt.close(fig)

    write_report(summary_rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(summary_rows: list[dict[str, object]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Learning Examples",
                "",
                "Estos tres pozos de entrenamiento tienen `TVT` real completo, pero simulamos el problema mirando el punto donde `TVT_input` se corta.",
                "",
                "En la figura:",
                "",
                "- La linea negra es el `TVT` real completo. En test no lo veremos.",
                "- La linea azul es `TVT_input`, la parte que Kaggle si nos deja ver.",
                "- La linea roja vertical es el corte: desde ahi empieza lo que hay que predecir.",
                "- La nube/curva de `GR` es la pista que puede ayudar a saber si el `TVT` sube, baja o queda plano.",
                "",
                "## Summary",
                "",
                markdown_table(summary_rows),
                "",
                "## Figure",
                "",
                "`reports/figures/learning_examples.png`",
                "",
            ]
        )
    )


if __name__ == "__main__":
    plot_examples()
