"""
Exploratory data analysis for the Mallorn astronomical classification dataset.
Run with an activated virtual environment:
    python eda_mallorn.py
Plots and tabular summaries are written to eda_outputs/.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure plotting defaults.
sns.set_theme(style="whitegrid")

DATA_DIR = Path(".")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_log.csv"
TEST_PATH = DATA_DIR / "test_log.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV with common NA sentinels normalized to NaN."""
    return pd.read_csv(path, na_values=["", " "])


def save_fig(fig: plt.Figure, name: str) -> None:
    """Persist a matplotlib figure and close it to free memory."""
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def summarize_dataframe(df: pd.DataFrame, name: str) -> None:
    """Write basic shape, dtypes, and missingness summaries to disk."""
    info_path = OUTPUT_DIR / f"{name}_info.txt"
    missing_path = OUTPUT_DIR / f"{name}_missing.csv"

    buffer = []
    buffer.append(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    buffer.append("\nColumn types:")
    buffer.append(df.dtypes.to_string())

    missing = df.isna().mean().sort_values(ascending=False)
    missing.to_csv(missing_path, header=["missing_rate"])
    buffer.append("\nMissingness saved to: " + str(missing_path))

    info_path.write_text("\n".join(buffer))


def plot_target_distribution(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    # Palette requires hue in seaborn >=0.14; mirror x into hue and hide legend.
    sns.countplot(data=train, x="target", hue="target", palette="viridis", legend=False, ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel("target")
    ax.set_ylabel("count")
    save_fig(fig, "target_distribution")


def plot_spec_type(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = train["SpecType"].value_counts(dropna=False).index
    sns.countplot(data=train, y="SpecType", order=order, hue="target", palette="crest", ax=ax)
    ax.set_title("SpecType by Target")
    ax.set_xlabel("count")
    ax.set_ylabel("SpecType")
    save_fig(fig, "spec_type_by_target")


def plot_split_balance(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=train, x="split", hue="target", palette="flare", ax=ax)
    ax.set_title("Split Balance and Target Mix")
    ax.set_xlabel("split")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "split_balance")


def plot_numeric_distributions(train: pd.DataFrame, numeric_cols: Iterable[str]) -> None:
    for col in numeric_cols:
        subset = train[[col, "target"]].dropna()
        if subset.empty:
            continue
        enable_kde = len(subset[col]) > 1
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=subset,
            x=col,
            hue="target",
            kde=enable_kde,
            element="step",
            stat="density",
            palette="rocket",
            ax=ax,
        )
        ax.set_title(f"Distribution of {col}")
        save_fig(fig, f"dist_{col}")


def plot_correlation(train: pd.DataFrame, numeric_cols: Iterable[str]) -> None:
    corr = train[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Correlation")
    save_fig(fig, "correlation_numeric")


def plot_scatter_pairs(train: pd.DataFrame, x: str, y: str) -> None:
    subset = train[[x, y, "target"]].dropna()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=subset, x=x, y=y, hue="target", palette="mako", alpha=0.7, ax=ax)
    ax.set_title(f"{x} vs {y}")
    save_fig(fig, f"scatter_{x}_vs_{y}")


def translation_length_features(df: pd.DataFrame) -> pd.Series:
    return df["English Translation"].fillna("").str.len()


def plot_translation_length(train: pd.DataFrame) -> None:
    train = train.copy()
    train["translation_len"] = translation_length_features(train)
    subset = train.dropna(subset=["translation_len"])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=subset,
        x="translation_len",
        hue="target",
        bins=40,
        element="step",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Length of English Translation (characters)")
    save_fig(fig, "translation_length")


def top_translation_tokens(df: pd.DataFrame, n: int = 25) -> Counter:
    tokens = []
    for text in df["English Translation"].dropna():
        tokens.extend(re.findall(r"[A-Za-z']+", str(text).lower()))
    counter = Counter(tokens)
    (OUTPUT_DIR / "top_translation_tokens.csv").write_text(
        "token,count\n" + "\n".join(f"{tok},{cnt}" for tok, cnt in counter.most_common(n))
    )
    return counter


def main() -> None:
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)

    summarize_dataframe(train, "train")
    summarize_dataframe(test, "test")

    plot_target_distribution(train)
    plot_spec_type(train)
    plot_split_balance(train)

    numeric_cols = [col for col in ["Z", "Z_err", "EBV"] if col in train.columns]
    plot_numeric_distributions(train, numeric_cols)
    if len(numeric_cols) > 1:
        plot_correlation(train, numeric_cols)
    if "Z" in train.columns and "EBV" in train.columns:
        plot_scatter_pairs(train, "Z", "EBV")
    if "Z" in train.columns and "Z_err" in train.columns:
        plot_scatter_pairs(train, "Z", "Z_err")

    plot_translation_length(train)
    top_translation_tokens(train)

    # Summary CSVs for quick inspection.
    train.describe(include="all").to_csv(OUTPUT_DIR / "train_describe.csv")
    test.describe(include="all").to_csv(OUTPUT_DIR / "test_describe.csv")

    print("EDA artifacts written to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
