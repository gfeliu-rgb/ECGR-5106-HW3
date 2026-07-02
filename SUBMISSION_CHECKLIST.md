# Homework 3 Submission Checklist

Before submitting on Canvas, verify every item below.

- The first page of the PDF report shows full name, student ID, and `Homework 3`.
- The PDF report contains a working public GitHub hyperlink to the exact `Homework_3` repository/folder.
- `data/vast_english_french.txt` was used as the only dataset.
- `results/split_indices.json` exists and confirms one deterministic 80/20 split reused across all problems.
- Problem 1 section includes baseline GRU architecture, training loss curve, validation loss curve, exact sequence accuracy, corpus BLEU-4, and 3-5 qualitative validation translations.
- Problem 2 section includes attention architecture, training/validation loss curves, exact sequence accuracy, corpus BLEU-4, comparison against Problem 1, and 1-2 attention maps.
- Problem 3 section includes French-to-English baseline and attention experiments, training/validation loss curves for both, exact sequence accuracies, BLEU-4 scores, qualitative samples, and a final direction-comparison discussion.
- `results/summary.csv` contains four rows: baseline EN-FR, attention EN-FR, baseline FR-EN, attention FR-EN.
- `results/*_samples.csv` files contain source, reference, prediction, exact-match status, and sentence BLEU-4.
- `plots/*_loss.png` contains one loss figure for each model/direction.
- `plots/*attention*.png` contains attention visualizations for attention models.
- The GitHub repository includes clean source code, README, requirements, report files, generated figures/tables, and executed notebook if you choose to add one.
- Open the GitHub link in a private/incognito browser window and confirm it is publicly accessible.
