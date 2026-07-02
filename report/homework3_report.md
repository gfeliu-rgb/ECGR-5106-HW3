# Homework 3: Sequence-to-Sequence Machine Translation

**Name:** Gilberto Feliu  
**Student ID:** 801257813  
**Assignment:** Homework 3  
**GitHub Repository:** https://github.com/gfeliu-rgb/ECGR-5106-HW3

## Dataset and Experimental Setup

The provided `vast_english_french.txt` dataset contains 555 English/French sentence pairs. I used one deterministic 80/20 split with random seed `4106`, giving 444 training pairs and 111 validation pairs. I saved the split indices in `results/split_indices.json` so I could verify that the English-to-French and French-to-English experiments were evaluated on the same underlying examples.

Text was normalized by lowercasing, removing accent marks, separating punctuation, and collapsing whitespace. Each vocabulary included `<pad>`, `<sos>`, `<eos>`, and `<unk>` tokens. Models were trained with token-level cross-entropy loss, Adam optimization, gradient clipping, teacher forcing during training, and greedy decoding during validation.

The two rubric metrics are traditional exact sequence accuracy and corpus-level word BLEU-4. I also tracked character BLEU-4, character accuracy, and sequence similarity because exact match alone was too strict to explain the partial translations.

## Problem 1: Baseline GRU Encoder-Decoder

### Architecture

The baseline English-to-French model uses a GRU encoder and GRU decoder. The encoder embeds the source English sentence and processes it with a GRU. The final encoder hidden state initializes the decoder. The decoder then predicts one French token at a time, using teacher forcing during training and greedy autoregressive decoding during validation.

Architecture assumptions:

- Source vocabulary size: 901
- Target vocabulary size: 996
- Embedding size: 128
- Hidden size: 160
- Batch size: 96
- Epochs: 60

### Loss Curves

Training and validation cross-entropy losses are plotted in `plots/problem1_baseline_en_fr_loss.png`. The best validation loss was **5.2227**.

### Validation Metrics

- Traditional sequence accuracy: **0.0000**
- Validation word BLEU-4: **0.0276**
- Character BLEU-4: **0.1463**
- Character accuracy: **0.1483**
- Sequence similarity: **0.4289**

### Qualitative Validation

The sample translations in `results/problem1_baseline_en_fr_samples.csv` show the main weakness of the baseline. It learned some frequent French sentence patterns, but it often fell back to common fragments instead of producing the full target sentence. For example, for `they visit museums often`, the target was `ils visitent souvent des musees`, while the model predicted `ils parlent souvent de`. That prediction is not correct, but it shares enough structure to produce nonzero BLEU and sequence-similarity scores.

| English source | Target French | Predicted French | Exact | BLEU-4 |
|---|---|---|---:|---:|
| i can see a large cruise ship in the distance | je peux voir un grand navire de croisiere au loin | je veux une une de de la | 0 | 0.1339 |
| she is looking for her lost dog | elle cherche son chien perdu | elle a des des pour | 0 | 0.2730 |
| he finished his homework quickly | il a fini ses devoirs rapidement | il a la chimie de | 0 | 0.2942 |
| we prefer walking through the quiet botanical gardens | nous preferons nous promener dans les jardins botaniques calmes | nous avons visite le lever de la | 0 | 0.1396 |
| they visit museums often | ils visitent souvent des musees | ils parlent souvent de | 0 | 0.3097 |

## Problem 2: GRU Encoder-Decoder with Luong Attention

### Architecture

The attention model uses the same GRU encoder-decoder backbone as Problem 1, but adds Luong dot-product attention. At each decoder step, the current decoder hidden state scores every encoder output. The resulting attention distribution forms a context vector, allowing the decoder to use all source positions instead of relying only on the final encoder hidden state.

Architecture assumptions:

- Source vocabulary size: 901
- Target vocabulary size: 996
- Embedding size: 128
- Hidden size: 160
- Batch size: 96
- Epochs: 60

### Loss Curves

Training and validation cross-entropy losses are plotted in `plots/problem2_attention_en_fr_loss.png`. The best validation loss was **5.0146**.

### Validation Metrics

- Traditional sequence accuracy: **0.0000**
- Validation word BLEU-4: **0.0589**
- Character BLEU-4: **0.2431**
- Character accuracy: **0.1835**
- Sequence similarity: **0.4688**

### Comparative Analysis

Attention improved the English-to-French model on the main BLEU-4 metric: word BLEU-4 increased from **0.0276** to **0.0589**. The best validation loss also improved from **5.2227** to **5.0146**. Traditional exact sequence accuracy stayed at **0.0000** for both models. This happened because the validation sentences require a full word-for-word match; a sentence can contain several useful translated words and still fail exact match because one word is missing, repeated, or in the wrong position.

The attention maps in `plots/problem2_attention_en_fr_attention_1.png` and `plots/problem2_attention_en_fr_attention_2.png` visualize the decoder's source-token focus during generation.

| English source | Target French | Predicted French | Exact | BLEU-4 |
|---|---|---|---:|---:|
| i can see a large cruise ship in the distance | je peux voir un grand navire de croisiere au loin | ils ont dans un navire au | 0 | 0.1349 |
| she is looking for her lost dog | elle cherche son chien perdu | elle cherche des contrats pour | 0 | 0.3593 |
| he finished his homework quickly | il a fini ses devoirs rapidement | il ecrit des scenarios | 0 | 0.2179 |
| we prefer walking through the quiet botanical gardens | nous preferons nous promener dans les jardins botaniques calmes | nous avons traverse la sur le parc | 0 | 0.1396 |
| they visit museums often | ils visitent souvent des musees | ils ont de la | 0 | 0.2798 |

The attention maps were not perfectly sharp. I interpret them as qualitative evidence that the model used source-position information, not as proof of clean word-by-word alignment. This matches the metric results: attention improved BLEU-4 and validation loss, but the generated sentences still contained repeated or misplaced high-frequency phrases.

## Problem 3: Reversed French-to-English Translation

### Baseline GRU Architecture

For the reversed task, the same baseline GRU encoder-decoder was trained with French as the source language and English as the target language. The same train/validation split was reused, but each pair was reversed.

The loss curve is saved as `plots/problem3_baseline_fr_en_loss.png`. The best validation loss was **4.8302**.

Validation metrics:

- Traditional sequence accuracy: **0.0000**
- Validation word BLEU-4: **0.0249**
- Character BLEU-4: **0.1414**
- Character accuracy: **0.1675**
- Sequence similarity: **0.4229**

Qualitative samples are saved in `results/problem3_baseline_fr_en_samples.csv`.

| French source | Target English | Predicted English | Exact | BLEU-4 |
|---|---|---|---:|---:|
| je peux voir un grand navire de croisiere au loin | i can see a large cruise ship in the distance | i want a a a the of | 0 | 0.1439 |
| elle cherche son chien perdu | she is looking for her lost dog | she is looking for a | 0 | 0.5093 |
| il a fini ses devoirs rapidement | he finished his homework quickly | he writes his for a | 0 | 0.3021 |
| nous preferons nous promener dans les jardins botaniques calmes | we prefer walking through the quiet botanical gardens | we are going to the the | 0 | 0.1752 |
| ils visitent souvent des musees | they visit museums often | they often talk about | 0 | 0.3976 |

### Attention GRU Architecture

The reversed attention model used the same Luong attention mechanism as Problem 2, again with French as the source and English as the target.

The loss curve is saved as `plots/problem3_attention_fr_en_loss.png`. The best validation loss was **4.6855**.

Validation metrics:

- Traditional sequence accuracy: **0.0000**
- Validation word BLEU-4: **0.0771**
- Character BLEU-4: **0.2481**
- Character accuracy: **0.1955**
- Sequence similarity: **0.4697**

Qualitative samples are saved in `results/problem3_attention_fr_en_samples.csv`. Attention maps are saved in `plots/problem3_attention_fr_en_attention_1.png` and `plots/problem3_attention_fr_en_attention_2.png`.

| French source | Target English | Predicted English | Exact | BLEU-4 |
|---|---|---|---:|---:|
| je peux voir un grand navire de croisiere au loin | i can see a large cruise ship in the distance | i see a a a ship the | 0 | 0.1894 |
| elle cherche son chien perdu | she is looking for her lost dog | she is looking for her | 0 | 0.6703 |
| il a fini ses devoirs rapidement | he finished his homework quickly | he cleans his his his bicycle | 0 | 0.2445 |
| nous preferons nous promener dans les jardins botaniques calmes | we prefer walking through the quiet botanical gardens | we are going to the the the | 0 | 0.1782 |
| ils visitent souvent des musees | they visit museums often | they often talk about modern | 0 | 0.3021 |

### Synthesis Discussion

In this run, the attention model was the strongest model in both translation directions. For English-to-French, attention improved word BLEU-4 from **0.0276** to **0.0589** and reduced best validation loss from **5.2227** to **5.0146**. For French-to-English, attention improved word BLEU-4 from **0.0249** to **0.0771** and reduced best validation loss from **4.8302** to **4.6855**.

French-to-English appeared easier for the attention model to optimize because it achieved the best overall validation loss (**4.6855**) and the highest word BLEU-4 (**0.0771**). Exact-match accuracy remained **0.0000** across all models, but this does not mean the models learned nothing. The qualitative examples and BLEU/sequence-similarity values show partial translations and repeated phrase patterns. The strict exact-match metric penalizes any missing word, word-order difference, or synonym, while BLEU-4 captures partial n-gram overlap.

## Conclusion

The baseline GRU encoder-decoder produced a working translation pipeline, but the generated sentences were usually incomplete or mixed with high-frequency phrase fragments. Adding Luong attention gave the decoder better access to the source sentence and improved validation loss, BLEU-4, character BLEU-4, and sequence similarity in both directions. The reversed French-to-English attention model performed best overall by validation loss and BLEU-4, so I concluded that this direction was slightly easier for this network and dataset split.
