"""
Utility functions: EDA visualizations + misc helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from pathlib import Path


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PALETTE = {0: "#4C72B0", 1: "#DD5A5A"}   # blue = non-toxic, red = toxic
LABEL_NAMES = {0: "Non-toxic", 1: "Toxic"}

def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label",
                             save_path: str = None) -> None:
    """Bar + pie chart of class distribution, highlighting the imbalance."""
    counts = df[label_col].value_counts().sort_index()
    labels = [LABEL_NAMES[i] for i in counts.index]
    colors = [PALETTE[i] for i in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                     f"{val:,}\n({val/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=11)
    axes[0].set_title("Class Distribution (absolute)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, counts.max() * 1.18)

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Class Distribution (proportion)", fontsize=13, fontweight="bold")

    plt.suptitle("⚠  Severe Class Imbalance — 93 % Non-toxic vs 7 % Toxic",
                 fontsize=12, color="#DD5A5A", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_tweet_length_distribution(df: pd.DataFrame, text_col: str = "tweet",
                                    label_col: str = "label",
                                    save_path: str = None) -> None:
    """KDE + box plots of character and word lengths, split by class."""
    df = df.copy()
    df["char_len"] = df[text_col].str.len()
    df["word_len"] = df[text_col].str.split().str.len()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col_idx, (metric, title_suffix) in enumerate(
        [("char_len", "Character Length"), ("word_len", "Word Count")]
    ):
        # KDE by class
        for label, grp in df.groupby(label_col):
            sns.kdeplot(grp[metric], ax=axes[0][col_idx],
                        label=LABEL_NAMES[label], color=PALETTE[label], fill=True, alpha=0.35)
        axes[0][col_idx].set_title(f"Distribution of {title_suffix} by Class")
        axes[0][col_idx].set_xlabel(title_suffix)
        axes[0][col_idx].legend()

        # Box plot
        data_by_class = [df[df[label_col] == lbl][metric].values for lbl in sorted(df[label_col].unique())]
        bp = axes[1][col_idx].boxplot(data_by_class, patch_artist=True,
                                       labels=[LABEL_NAMES[l] for l in sorted(df[label_col].unique())])
        for patch, lbl in zip(bp["boxes"], sorted(df[label_col].unique())):
            patch.set_facecolor(PALETTE[lbl])
            patch.set_alpha(0.7)
        axes[1][col_idx].set_title(f"Box Plot — {title_suffix} by Class")
        axes[1][col_idx].set_ylabel(title_suffix)

    plt.suptitle("Tweet Length Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_top_words(df: pd.DataFrame, text_col: str = "tweet_clean",
                   label_col: str = "label", top_n: int = 20,
                   save_path: str = None) -> None:
    """
    Horizontal bar chart of top-N most frequent words, one panel per class.
    Requires the 'tweet_clean' column (preprocessed text).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, label in zip(axes, sorted(df[label_col].unique())):
        corpus = " ".join(df[df[label_col] == label][text_col].dropna())
        freq = Counter(corpus.split()).most_common(top_n)
        words, counts = zip(*freq)

        ax.barh(range(top_n), counts[::-1], color=PALETTE[label], alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(words[::-1], fontsize=10)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top {top_n} Words — {LABEL_NAMES[label]}", fontweight="bold")
        ax.invert_yaxis()

    plt.suptitle("Most Frequent Words per Class (after preprocessing)", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_wordclouds(df: pd.DataFrame, text_col: str = "tweet_clean",
                    label_col: str = "label", save_path: str = None) -> None:
    """Word cloud per class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, label in zip(axes, sorted(df[label_col].unique())):
        corpus = " ".join(df[df[label_col] == label][text_col].dropna())
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Reds" if label == 1 else "Blues",
            max_words=200,
            collocations=False,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {LABEL_NAMES[label]}", fontsize=13, fontweight="bold")

    plt.suptitle("Word Clouds per Class (after preprocessing)", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_overview(feature_shapes: dict, save_path: str = None) -> None:
    """
    Bar chart showing the dimensionality of each extracted feature set.
    feature_shapes: {'BoW': 10000, 'TF-IDF': 15000, ...}
    """
    names = list(feature_shapes.keys())
    dims = list(feature_shapes.values())
    colors = sns.color_palette("tab10", len(names))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, dims, color=colors, edgecolor="white")
    for bar, val in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dims) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Feature Dimensionality per Method", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of dimensions / features")
    ax.set_ylim(0, max(dims) * 1.15)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Summary statistics helpers
# ---------------------------------------------------------------------------

def eda_summary(df: pd.DataFrame, text_col: str = "tweet",
                label_col: str = "label") -> pd.DataFrame:
    """Return a text-length statistics table grouped by class."""
    df = df.copy()
    df["char_len"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()

    stats = (
        df.groupby(label_col)[["char_len", "word_count"]]
        .agg(["mean", "median", "std", "min", "max"])
        .round(1)
    )
    stats.index = [LABEL_NAMES[i] for i in stats.index]
    return stats


def vocabulary_stats(df: pd.DataFrame, text_col: str = "tweet_clean") -> dict:
    """Count unique tokens in the cleaned corpus."""
    all_tokens = " ".join(df[text_col].dropna()).split()
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
        "avg_tokens_per_tweet": round(len(all_tokens) / len(df), 1),
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str = None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
