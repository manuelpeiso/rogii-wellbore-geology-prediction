from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

RAW_DIR = Path("data/raw")
REPORT_PATH = Path("reports/data_overview.md")
FIG_DIR = Path("reports/figures")


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def df_to_md_table(df: pd.DataFrame) -> str:
    table = df.reset_index()
    table.columns = [str(col) for col in table.columns]
    return md_table(table.to_dict("records"), table.columns.tolist())


def well_id(path: Path) -> str:
    return path.name.split("__")[0].split(".")[0]


def scan_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal_rows = []
    typewell_rows = []

    for path in sorted((RAW_DIR / split).glob("*__horizontal_well.csv")):
        df = pd.read_csv(path)
        wid = well_id(path)
        row = {
            "split": split,
            "well_id": wid,
            "rows": len(df),
            "md_min": df["MD"].min(),
            "md_max": df["MD"].max(),
            "gr_missing": int(df["GR"].isna().sum()),
            "gr_missing_pct": 100 * df["GR"].isna().mean(),
            "tvt_input_missing": int(df["TVT_input"].isna().sum()),
            "tvt_input_missing_pct": 100 * df["TVT_input"].isna().mean(),
            "ps_row": int(df["TVT_input"].isna().idxmax()) if df["TVT_input"].isna().any() else None,
        }
        if "TVT" in df.columns:
            row["tvt_min"] = df["TVT"].min()
            row["tvt_max"] = df["TVT"].max()
            row["target_available"] = int(df["TVT"].notna().sum())
        else:
            row["tvt_min"] = np.nan
            row["tvt_max"] = np.nan
            row["target_available"] = 0
        horizontal_rows.append(row)

    for path in sorted((RAW_DIR / split).glob("*__typewell.csv")):
        df = pd.read_csv(path)
        row = {
            "split": split,
            "well_id": well_id(path),
            "rows": len(df),
            "tvt_min": df["TVT"].min(),
            "tvt_max": df["TVT"].max(),
            "gr_missing": int(df["GR"].isna().sum()),
            "geology_missing": int(df["Geology"].isna().sum()) if "Geology" in df.columns else None,
            "geology_labels": int(df["Geology"].dropna().nunique()) if "Geology" in df.columns else 0,
        }
        typewell_rows.append(row)

    return pd.DataFrame(horizontal_rows), pd.DataFrame(typewell_rows)


def describe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].describe().T[["min", "mean", "50%", "max"]].round(2)


def plot_example_well() -> None:
    train_paths = sorted((RAW_DIR / "train").glob("*__horizontal_well.csv"))
    if not train_paths:
        return

    path = train_paths[0]
    wid = well_id(path)
    horizontal = pd.read_csv(path)
    typewell = pd.read_csv(RAW_DIR / "train" / f"{wid}__typewell.csv")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(horizontal["MD"], horizontal["TVT"], label="TVT target", linewidth=1.5)
    axes[0].plot(horizontal["MD"], horizontal["TVT_input"], label="TVT_input", linewidth=1)
    axes[0].set_title(f"Horizontal well {wid}")
    axes[0].set_xlabel("MD")
    axes[0].set_ylabel("TVT")
    axes[0].legend()

    axes[1].plot(typewell["GR"], typewell["TVT"], label="typewell GR", linewidth=1)
    axes[1].scatter(horizontal["GR"], horizontal["TVT"], label="horizontal GR", s=3, alpha=0.35)
    axes[1].invert_yaxis()
    axes[1].set_title("GR signatures on TVT")
    axes[1].set_xlabel("GR")
    axes[1].set_ylabel("TVT")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_well_tvt_gr.png", dpi=160)
    plt.close(fig)


def plot_missing_distributions(horizontal: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = horizontal[horizontal["split"] == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train["rows"], bins=30)
    axes[0].set_title("Train horizontal rows per well")
    axes[0].set_xlabel("rows")
    axes[0].set_ylabel("wells")

    axes[1].hist(train["tvt_input_missing_pct"], bins=30)
    axes[1].set_title("Hidden TVT_input percentage")
    axes[1].set_xlabel("% missing")
    axes[1].set_ylabel("wells")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "train_size_and_hidden_pct.png", dpi=160)
    plt.close(fig)


def main() -> None:
    train_h, train_t = scan_split("train")
    test_h, test_t = scan_split("test")
    horizontal = pd.concat([train_h, test_h], ignore_index=True)
    typewell = pd.concat([train_t, test_t], ignore_index=True)

    sample = pd.read_csv(RAW_DIR / "sample_submission.csv")
    png_count = len(list((RAW_DIR / "train").glob("*.png")))
    all_geology = []
    for path in sorted((RAW_DIR / "train").glob("*__typewell.csv")):
        df = pd.read_csv(path)
        if "Geology" in df.columns:
            all_geology.append(df["Geology"].dropna())
    geology_counts = pd.concat(all_geology).value_counts() if all_geology else pd.Series(dtype=int)

    plot_example_well()
    plot_missing_distributions(horizontal)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Overview",
        "",
        "## Files",
        "",
        md_table(
            [
                {"item": "train horizontal CSVs", "count": len(train_h)},
                {"item": "train typewell CSVs", "count": len(train_t)},
                {"item": "train PNG images", "count": png_count},
                {"item": "test horizontal CSVs", "count": len(test_h)},
                {"item": "test typewell CSVs", "count": len(test_t)},
                {"item": "sample submission rows", "count": len(sample)},
            ],
            ["item", "count"],
        ),
        "",
        "## Horizontal Well Columns",
        "",
        "- Train: `MD`, `X`, `Y`, `Z`, geology top markers (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`), `TVT`, `GR`, `TVT_input`.",
        "- Test: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`.",
        "- Target: `tvt` in `sample_submission.csv`, keyed by `<well_id>_<row_index>`.",
        "",
        "## Horizontal Well Summary",
        "",
        df_to_md_table(describe_numeric(horizontal, ["rows", "gr_missing_pct", "tvt_input_missing_pct", "ps_row"])),
        "",
        "## Test Wells",
        "",
        md_table(
            test_h.round(2).to_dict("records"),
            ["well_id", "rows", "md_min", "md_max", "gr_missing", "tvt_input_missing", "ps_row"],
        ),
        "",
        "## Typewell Summary",
        "",
        df_to_md_table(describe_numeric(typewell, ["rows", "tvt_min", "tvt_max", "gr_missing", "geology_labels"])),
        "",
        "## Geology Labels In Train Typewells",
        "",
        md_table(
            [{"geology": str(k), "rows": int(v)} for k, v in geology_counts.items()],
            ["geology", "rows"],
        ),
        "",
        "## Figures",
        "",
        "- `reports/figures/example_well_tvt_gr.png`",
        "- `reports/figures/train_size_and_hidden_pct.png`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
