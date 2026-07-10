from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src" / "seq2seq_hw3.py"
PLOT_PATH = ROOT / "make_comparison_artifacts.py"
OUT_PATH = ROOT / "Homework_3_Seq2Seq_Machine_Translation.ipynb"


def extract_script_prelude(script_text: str) -> str:
    marker = "def parse_args():"
    idx = script_text.index(marker)
    return script_text[:idx].rstrip()


def extract_plot_functions(plot_text: str) -> str:
    marker = "def main():"
    idx = plot_text.index(marker)
    snippet = plot_text[:idx].rstrip()
    return snippet.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = Path.cwd()')


def main() -> None:
    seq2seq_text = SRC_PATH.read_text(encoding="utf-8")
    plot_text = PLOT_PATH.read_text(encoding="utf-8")

    training_defs = extract_script_prelude(seq2seq_text)
    plot_defs = extract_plot_functions(plot_text)

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}

    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                """
                # Homework 3: Sequence-to-Sequence Machine Translation

                **Name:** Gilberto Feliu  
                **Student ID:** 801257813  
                **Assignment:** Homework 3

                This notebook is the primary reproducible source for the Homework 3 report. It loads the dataset, defines the baseline and attention-based seq2seq models, trains all four required experiments, computes evaluation metrics, and regenerates the exact CSV and PNG artifacts referenced by the report.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                """
                ## Environment and Dataset Check

                The notebook expects the provided dataset at `data/vast_english_french.txt` and writes regenerated outputs into `results/` and `plots/`.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                from pathlib import Path

                ROOT = Path.cwd()
                DATA_PATH = ROOT / "data" / "vast_english_french.txt"
                RESULTS_DIR = ROOT / "results"
                PLOTS_DIR = ROOT / "plots"

                print("Working directory:", ROOT)
                print("Dataset path:", DATA_PATH)
                print("Dataset exists:", DATA_PATH.exists())
                if not DATA_PATH.exists():
                    raise FileNotFoundError("Place vast_english_french.txt in Homework_3/data/ before executing this notebook.")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Model and Utility Definitions\n\nThe following cell contains the full implementation used for the homework experiments."
        )
    )
    cells.append(nbf.v4.new_code_cell(training_defs))
    cells.append(
        nbf.v4.new_markdown_cell(
            textwrap.dedent(
                """
                ## Training Configuration

                These hyperparameters match the report configuration and regenerate the same artifact filenames used throughout the repository.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                class NotebookArgs:
                    data = "data/vast_english_french.txt"
                    epochs = 60
                    batch_size = 96
                    emb_dim = 128
                    hidden_dim = 160
                    dropout = 0.2
                    lr = 0.001
                    seed = 4106
                    min_freq = 1
                    max_vocab = None
                    max_decode_len = 40
                    cpu = False
                    quick = False


                args = NotebookArgs()
                args
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Run All Four Experiments\n\nThis cell performs the deterministic split, trains the models, writes `results/*.csv`, and generates the per-model loss and attention plots under `plots/`."
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                random.seed(args.seed)
                torch.manual_seed(args.seed)

                RESULTS_DIR.mkdir(exist_ok=True)
                PLOTS_DIR.mkdir(exist_ok=True)

                pairs = load_pairs(DATA_PATH)
                train_pairs, val_pairs, train_idx, val_idx = split_pairs(pairs, seed=args.seed)

                with (RESULTS_DIR / "split_indices.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {"seed": args.seed, "train_indices": train_idx, "validation_indices": val_idx},
                        f,
                        indent=2,
                    )

                experiments = [
                    ("problem1_baseline_en_fr", False, False),
                    ("problem2_attention_en_fr", False, True),
                    ("problem3_baseline_fr_en", True, False),
                    ("problem3_attention_fr_en", True, True),
                ]

                summaries = []
                for name, reverse, attention in experiments:
                    summaries.append(
                        train_one_experiment(
                            name,
                            train_pairs,
                            val_pairs,
                            RESULTS_DIR,
                            PLOTS_DIR,
                            args,
                            reverse=reverse,
                            attention=attention,
                        )
                    )

                with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                    writer.writeheader()
                    writer.writerows(summaries)

                print("\\nFinal validation summary")
                for row in summaries:
                    print(
                        f"{row['experiment']}: val_loss={row['best_val_loss']:.4f}, "
                        f"exact={row['exact_match']:.4f}, word BLEU-4={row['word_bleu4']:.4f}"
                    )
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Comparison Plots\n\nThe next cell regenerates the same comparison figures referenced by the report using the freshly written `results/summary.csv` and per-experiment history files."
        )
    )
    cells.append(nbf.v4.new_code_cell(plot_defs))
    cells.append(
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                ROOT = Path.cwd()
                RESULTS = ROOT / "results"
                PLOTS = ROOT / "plots"
                SUMMARY = pd.read_csv(RESULTS / "summary.csv")

                for experiment in EXPERIMENT_LABELS:
                    improve_loss_plot(experiment)
                make_problem2_comparison()
                make_problem3_comparison()

                print("Updated loss plots and comparison artifacts.")
                """
            ).strip()
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Summary Metrics"))
    cells.append(
        nbf.v4.new_code_cell(
            "summary = pd.read_csv('results/summary.csv')\nsummary[['experiment', 'direction', 'architecture', 'best_val_loss', 'exact_match', 'word_bleu4', 'char_bleu4', 'sequence_similarity']]"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Qualitative Samples"))
    cells.append(
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                for path in sorted(Path('results').glob('*_samples.csv')):
                    print(f"\\n{path.name}")
                    display(pd.read_csv(path))
                """
            ).strip()
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Report Figures"))
    cells.append(
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                from IPython.display import Image, display

                for path in sorted(Path('plots').glob('*.png')):
                    print(path.name)
                    display(Image(filename=str(path)))
                """
            ).strip()
        )
    )

    nb.cells = cells
    nbf.write(nb, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
