import argparse
import csv
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


def normalize_text(text):
    text = unicodedata.normalize("NFD", text.lower().strip())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"([.!?;,])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?;,']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pairs(path):
    pairs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                left, right = line.split("\t")[:2]
            elif "|||" in line:
                left, right = line.split("|||")[:2]
            else:
                raise ValueError(
                    "Expected each dataset row to contain tab-separated or |||-separated English/French text."
                )
            pairs.append((normalize_text(left), normalize_text(right)))
    if len(pairs) < 10:
        raise ValueError("Dataset is too small after parsing.")
    return pairs


def split_pairs(pairs, seed=4106, train_fraction=0.8):
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    cutoff = int(len(indices) * train_fraction)
    train_idx = indices[:cutoff]
    val_idx = indices[cutoff:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx], train_idx, val_idx


class Vocab:
    def __init__(self, texts, min_freq=1, max_size=None):
        counts = Counter()
        for text in texts:
            counts.update(text.split())
        words = [w for w, c in counts.items() if c >= min_freq]
        words.sort(key=lambda w: (-counts[w], w))
        if max_size:
            words = words[: max(0, max_size - len(SPECIALS))]
        self.itos = SPECIALS + words
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.sos_idx = self.stoi[SOS]
        self.eos_idx = self.stoi[EOS]
        self.unk_idx = self.stoi[UNK]

    def encode(self, text, add_boundaries=True):
        ids = [self.stoi.get(tok, self.unk_idx) for tok in text.split()]
        if add_boundaries:
            return [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids):
        words = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_idx:
                break
            if idx in (self.pad_idx, self.sos_idx):
                continue
            words.append(self.itos[idx] if idx < len(self.itos) else UNK)
        return " ".join(words)

    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.rows = [
            (
                torch.tensor(src_vocab.encode(src), dtype=torch.long),
                torch.tensor(tgt_vocab.encode(tgt), dtype=torch.long),
                src,
                tgt,
            )
            for src, tgt in pairs
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_batch(rows):
    src, tgt, src_text, tgt_text = zip(*rows)
    src_lens = torch.tensor([len(x) for x in src], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgt], dtype=torch.long)
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src_pad, src_lens, tgt_pad, tgt_lens, list(src_text), list(tgt_text)


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class BaselineDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, encoder_outputs=None, src_mask=None):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        output, hidden = self.gru(embedded, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden, None


class LuongAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, encoder_outputs, src_mask):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        scores = torch.bmm(encoder_outputs, hidden[-1].unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs)
        gru_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        logits = self.out(torch.cat([output.squeeze(1), context.squeeze(1)], dim=1))
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=128, hidden_dim=256, dropout=0.2, attention=False):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, dropout)
        decoder_cls = LuongAttentionDecoder if attention else BaselineDecoder
        self.decoder = decoder_cls(tgt_vocab_size, emb_dim, hidden_dim, dropout)
        self.attention = attention

    def forward(self, src, src_lens, tgt, teacher_forcing=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)
        encoder_outputs, hidden = self.encoder(src, src_lens)
        src_mask = torch.arange(src.shape[1], device=src.device).unsqueeze(0) < src_lens.unsqueeze(1).to(src.device)
        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.forward_step(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t - 1] = logits
            use_teacher = random.random() < teacher_forcing
            input_token = tgt[:, t] if use_teacher else logits.argmax(1)
        return outputs

    @torch.no_grad()
    def generate(self, src, src_lens, sos_idx, eos_idx, max_len=40):
        self.eval()
        encoder_outputs, hidden = self.encoder(src, src_lens)
        src_mask = torch.arange(src.shape[1], device=src.device).unsqueeze(0) < src_lens.unsqueeze(1).to(src.device)
        input_token = torch.full((src.shape[0],), sos_idx, dtype=torch.long, device=src.device)
        sequences = []
        attentions = []
        finished = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, hidden, attn = self.decoder.forward_step(input_token, hidden, encoder_outputs, src_mask)
            input_token = logits.argmax(1)
            sequences.append(input_token)
            if attn is not None:
                attentions.append(attn)
            finished |= input_token.eq(eos_idx)
            if bool(finished.all()):
                break
        pred = torch.stack(sequences, dim=1) if sequences else torch.empty(src.shape[0], 0, device=src.device)
        attn_tensor = torch.stack(attentions, dim=1) if attentions else None
        return pred, attn_tensor


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference, hypothesis, max_n=4):
    ref = reference.split()
    hyp = hypothesis.split()
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = ngram_counts(hyp, n)
        ref_counts = ngram_counts(ref, n)
        total = sum(hyp_counts.values())
        overlap = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append((overlap + 1.0) / (total + 1.0))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(hyp)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def corpus_bleu(references, hypotheses, max_n=4, tokenizer=None):
    if not hypotheses:
        return 0.0
    tokenizer = tokenizer or (lambda text: text.split())
    matches = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    hyp_len = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref = tokenizer(reference)
        hyp = tokenizer(hypothesis)
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            hyp_counts = ngram_counts(hyp, n)
            ref_counts = ngram_counts(ref, n)
            matches[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            totals[n - 1] += sum(hyp_counts.values())
    precisions = [(matches[i] + 1.0) / (totals[i] + 1.0) for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def char_accuracy(reference, hypothesis):
    if not reference:
        return 1.0 if not hypothesis else 0.0
    matches = sum(1 for a, b in zip(reference, hypothesis) if a == b)
    return matches / len(reference)


def sequence_similarity(reference, hypothesis):
    return SequenceMatcher(None, reference, hypothesis).ratio()


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    for src, src_lens, tgt, _tgt_lens, *_ in loader:
        src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(src, src_lens, tgt, teacher_forcing=0.5 if train else 0.0)
            gold = tgt[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), gold.reshape(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        tokens = gold.ne(0).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_generation(model, loader, src_vocab, tgt_vocab, device, max_len=40):
    predictions, references, sources = [], [], []
    sample_attn = []
    for src, src_lens, _tgt, _tgt_lens, src_text, tgt_text in loader:
        src, src_lens = src.to(device), src_lens.to(device)
        pred_ids, attn = model.generate(src, src_lens, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=max_len)
        for row in pred_ids.cpu():
            predictions.append(tgt_vocab.decode(row.tolist()))
        references.extend(tgt_text)
        sources.extend(src_text)
        if attn is not None and len(sample_attn) < 2:
            for b in range(min(src.shape[0], 2 - len(sample_attn))):
                pred_tokens = predictions[-src.shape[0] + b].split()
                src_tokens = sources[-src.shape[0] + b].split()
                sample_attn.append(
                    {
                        "source": src_tokens,
                        "prediction": pred_tokens,
                        "weights": attn[b, : len(pred_tokens), : len(src_tokens)].cpu(),
                    }
                )
    exact = sum(p == r for p, r in zip(predictions, references)) / max(1, len(references))
    word_bleu = corpus_bleu(references, predictions)
    char_bleu = corpus_bleu(references, predictions, tokenizer=list)
    char_acc = sum(char_accuracy(r, p) for r, p in zip(references, predictions)) / max(1, len(references))
    seq_sim = sum(sequence_similarity(r, p) for r, p in zip(references, predictions)) / max(1, len(references))
    return {
        "sources": sources,
        "references": references,
        "predictions": predictions,
        "exact_match": exact,
        "word_bleu4": word_bleu,
        "char_bleu4": char_bleu,
        "char_accuracy": char_acc,
        "sequence_similarity": seq_sim,
        "attention_samples": sample_attn,
    }


def save_loss_plot(history, out_path, title):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_attention_map(sample, out_path, title):
    weights = sample["weights"]
    if weights.numel() == 0:
        return
    plt.figure(figsize=(max(5, len(sample["source"]) * 0.55), max(4, len(sample["prediction"]) * 0.45)))
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xticks(range(len(sample["source"])), sample["source"], rotation=45, ha="right")
    plt.yticks(range(len(sample["prediction"])), sample["prediction"])
    plt.xlabel("Source tokens")
    plt.ylabel("Predicted target tokens")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_history_csv(history, path):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({k: history[k][i] for k in writer.fieldnames})


def write_samples_csv(eval_result, path, limit=5):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        fields = [
            "source",
            "reference",
            "prediction",
            "exact_match",
            "sentence_bleu4",
            "char_accuracy",
            "sequence_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for src, ref, pred in zip(eval_result["sources"][:limit], eval_result["references"][:limit], eval_result["predictions"][:limit]):
            writer.writerow(
                {
                    "source": src,
                    "reference": ref,
                    "prediction": pred,
                    "exact_match": int(pred == ref),
                    "sentence_bleu4": f"{sentence_bleu(ref, pred):.4f}",
                    "char_accuracy": f"{char_accuracy(ref, pred):.4f}",
                    "sequence_similarity": f"{sequence_similarity(ref, pred):.4f}",
                }
            )


def train_one_experiment(name, train_pairs, val_pairs, results_dir, plots_dir, args, reverse=False, attention=False):
    if reverse:
        train_pairs = [(fr, en) for en, fr in train_pairs]
        val_pairs = [(fr, en) for en, fr in val_pairs]
    src_vocab = Vocab([src for src, _ in train_pairs], min_freq=args.min_freq, max_size=args.max_vocab)
    tgt_vocab = Vocab([tgt for _, tgt in train_pairs], min_freq=args.min_freq, max_size=args.max_vocab)
    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), args.emb_dim, args.hidden_dim, args.dropout, attention).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{name}: epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    eval_result = evaluate_generation(model, val_loader, src_vocab, tgt_vocab, device, max_len=args.max_decode_len)
    history_path = results_dir / f"{name}_history.csv"
    samples_path = results_dir / f"{name}_samples.csv"
    write_history_csv(history, history_path)
    write_samples_csv(eval_result, samples_path, limit=5)
    save_loss_plot(history, plots_dir / f"{name}_loss.png", name.replace("_", " ").title())
    if attention:
        for i, sample in enumerate(eval_result["attention_samples"], start=1):
            save_attention_map(sample, plots_dir / f"{name}_attention_{i}.png", f"{name} attention sample {i}")
    return {
        "experiment": name,
        "direction": "French-to-English" if reverse else "English-to-French",
        "architecture": "GRU with Luong attention" if attention else "Baseline GRU encoder-decoder",
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "source_vocab": len(src_vocab),
        "target_vocab": len(tgt_vocab),
        "best_val_loss": best_val,
        "exact_match": eval_result["exact_match"],
        "word_bleu4": eval_result["word_bleu4"],
        "char_bleu4": eval_result["char_bleu4"],
        "char_accuracy": eval_result["char_accuracy"],
        "sequence_similarity": eval_result["sequence_similarity"],
        "seconds": time.time() - start,
        "history_csv": str(history_path),
        "samples_csv": str(samples_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/vast_english_french.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab", type=int, default=None)
    parser.add_argument("--max-decode-len", type=int, default=40)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run fewer epochs on a smaller subset for smoke testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.hidden_dim = min(args.hidden_dim, 128)
        args.emb_dim = min(args.emb_dim, 64)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    results_dir = root / "results"
    plots_dir = root / "plots"
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    pairs = load_pairs(data_path)
    if args.quick:
        pairs = pairs[: min(len(pairs), 1200)]
    train_pairs, val_pairs, train_idx, val_idx = split_pairs(pairs, seed=args.seed)
    with (results_dir / "split_indices.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "train_indices": train_idx, "validation_indices": val_idx}, f, indent=2)
    experiments = [
        ("problem1_baseline_en_fr", False, False),
        ("problem2_attention_en_fr", False, True),
        ("problem3_baseline_fr_en", True, False),
        ("problem3_attention_fr_en", True, True),
    ]
    summaries = []
    for name, reverse, attention in experiments:
        summaries.append(train_one_experiment(name, train_pairs, val_pairs, results_dir, plots_dir, args, reverse, attention))
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print("\nFinal validation summary")
    for row in summaries:
        print(
            f"{row['experiment']}: val_loss={row['best_val_loss']:.4f}, "
            f"exact={row['exact_match']:.4f}, word BLEU-4={row['word_bleu4']:.4f}"
        )


if __name__ == "__main__":
    main()
