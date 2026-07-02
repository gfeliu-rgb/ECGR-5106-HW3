# ECGR 5106 / 4106 - Homework 3

Original sequence-to-sequence machine translation implementation for English/French translation with baseline GRU and attention-augmented GRU models.

## Files

- `src/seq2seq_hw3.py`: complete PyTorch training, evaluation, BLEU-4, exact-match, plotting, and attention-map code.
- `data/vast_english_french.txt`: place the provided assignment dataset here. This file is not included in the repository unless your instructor permits it.
- `results/`: generated CSV metrics, qualitative samples, and deterministic split indices.
- `plots/`: generated training/validation loss curves and attention visualizations.
- `report/homework3_report_template.md`: report template aligned to every assignment requirement.
- `SUBMISSION_CHECKLIST.md`: final Canvas/GitHub verification checklist.

## How To Run

```bash
pip install -r requirements.txt
python src/seq2seq_hw3.py --data data/vast_english_french.txt --epochs 10
```

For a fast smoke test after placing the dataset:

```bash
python src/seq2seq_hw3.py --data data/vast_english_french.txt --quick
```

The script uses one deterministic 80/20 split with seed `4106` and reuses that split for all four experiments:

1. Problem 1 baseline GRU, English-to-French.
2. Problem 2 GRU with Luong attention, English-to-French.
3. Problem 3 baseline GRU, French-to-English.
4. Problem 3 GRU with Luong attention, French-to-English.

## Submission Notes

The PDF report must include your name, student ID, assignment number, and a working public GitHub link to this `Homework_3` folder. Before submitting, run all experiments and verify that `results/summary.csv`, the four loss plots, the sample CSV files, and attention maps are present.
