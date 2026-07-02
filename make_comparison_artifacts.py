from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
SUMMARY = pd.read_csv(RESULTS / "summary.csv")


EXPERIMENT_LABELS = {
    "problem1_baseline_en_fr": "Problem 1 Baseline GRU",
    "problem2_attention_en_fr": "Problem 2 Attention GRU",
    "problem3_baseline_fr_en": "Problem 3 Baseline GRU",
    "problem3_attention_fr_en": "Problem 3 Attention GRU",
}

LOSS_TITLES = {
    "problem1_baseline_en_fr": "Problem 1: Baseline GRU English-to-French Loss",
    "problem2_attention_en_fr": "Problem 2: Attention GRU English-to-French Loss",
    "problem3_baseline_fr_en": "Problem 3: Baseline GRU French-to-English Loss",
    "problem3_attention_fr_en": "Problem 3: Attention GRU French-to-English Loss",
}


def improve_loss_plot(experiment):
    history = pd.read_csv(RESULTS / f"{experiment}_history.csv")
    best_idx = history["val_loss"].idxmin()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_val = float(history.loc[best_idx, "val_loss"])

    plt.figure(figsize=(9, 6))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation Loss")
    plt.axvline(best_epoch, color="red", linestyle="--", linewidth=2, label=f"Best Val Epoch ({best_epoch})")
    plt.scatter([best_epoch], [best_val], color="red", zorder=5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(LOSS_TITLES[experiment])
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"{experiment}_loss.png", dpi=180)
    plt.close()


def make_problem2_comparison():
    rows = SUMMARY[SUMMARY["experiment"].isin(["problem1_baseline_en_fr", "problem2_attention_en_fr"])].copy()
    rows["model"] = rows["experiment"].map(EXPERIMENT_LABELS)
    out = rows[
        [
            "model",
            "best_val_loss",
            "exact_match",
            "word_bleu4",
            "char_bleu4",
            "char_accuracy",
            "sequence_similarity",
            "seconds",
        ]
    ]
    out.to_csv(RESULTS / "problem1_vs_problem2_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["Baseline", "Attention"]
    axes[0].bar(labels, out["word_bleu4"], color=["#4c78a8", "#f58518"])
    axes[0].set_title("English-to-French BLEU-4")
    axes[0].set_ylabel("Word BLEU-4")
    axes[1].bar(labels, out["best_val_loss"], color=["#4c78a8", "#f58518"])
    axes[1].set_title("English-to-French Validation Loss")
    axes[1].set_ylabel("Best Cross-Entropy Loss")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS / "problem1_vs_problem2_comparison.png", dpi=180)
    plt.close()


def make_problem3_comparison():
    rows = SUMMARY.copy()
    rows["model"] = rows["experiment"].map(EXPERIMENT_LABELS)
    rows[
        [
            "model",
            "direction",
            "best_val_loss",
            "exact_match",
            "word_bleu4",
            "char_bleu4",
            "char_accuracy",
            "sequence_similarity",
            "seconds",
        ]
    ].to_csv(RESULTS / "problem3_full_direction_comparison.csv", index=False)

    ordered = [
        "problem1_baseline_en_fr",
        "problem2_attention_en_fr",
        "problem3_baseline_fr_en",
        "problem3_attention_fr_en",
    ]
    rows = rows.set_index("experiment").loc[ordered].reset_index()
    labels = ["EN-FR\nBaseline", "EN-FR\nAttention", "FR-EN\nBaseline", "FR-EN\nAttention"]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, rows["word_bleu4"], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    plt.ylabel("Word BLEU-4")
    plt.title("Validation BLEU-4 by Direction and Architecture")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS / "problem3_full_bleu_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, rows["best_val_loss"], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    plt.ylabel("Best Cross-Entropy Loss")
    plt.title("Best Validation Loss by Direction and Architecture")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS / "problem3_model_comparison.png", dpi=180)
    plt.close()


def main():
    for experiment in EXPERIMENT_LABELS:
        improve_loss_plot(experiment)
    make_problem2_comparison()
    make_problem3_comparison()
    print("Updated loss plots and comparison artifacts.")


if __name__ == "__main__":
    main()
