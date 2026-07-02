# ECGR 5106 / 4106 - Homework 3

Sequence-to-sequence machine translation implementation for the Homework 3 English/French translation assignment. The repository contains the code, executed notebook, dataset copy, generated results, figures, and final report used for the Canvas submission.

## Files

- `src/seq2seq_hw3.py`: complete PyTorch training, evaluation, BLEU-4, exact-match, plotting, and attention-map code.
- `data/vast_english_french.txt`: provided assignment dataset used for all experiments.
- `results/`: generated CSV metrics, qualitative samples, deterministic split indices, and comparison tables.
- `plots/`: generated training/validation loss curves, attention visualizations, and comparison plots.
- `report/homework3_report.pdf`: final report submitted to Canvas.
- `report/homework3_report.md`: Markdown source for the report text.
- `SUBMISSION_CHECKLIST.md`: final Canvas/GitHub verification checklist.

## How To Run

```bash
pip install -r requirements.txt
python src/seq2seq_hw3.py --data data/vast_english_french.txt --epochs 60 --batch-size 96 --hidden-dim 160 --emb-dim 128
```

For a fast smoke test:

```bash
python src/seq2seq_hw3.py --data data/vast_english_french.txt --quick
```

The script uses one deterministic 80/20 split with seed `4106` and reuses that split for all four experiments:

1. Problem 1 baseline GRU, English-to-French.
2. Problem 2 GRU with Luong attention, English-to-French.
3. Problem 3 baseline GRU, French-to-English.
4. Problem 3 GRU with Luong attention, French-to-English.

## Submission Notes

The PDF report includes the student information, assignment number, and public GitHub link. The experiments have already been run, and the executed notebook, `results/summary.csv`, the four loss plots, the sample CSV files, and the attention maps are included so the report values can be checked directly against the code outputs.
