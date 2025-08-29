import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

# =============== Repro ===================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============== Data ====================
class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_vocab, output_vocab):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.sos = output_vocab['<sos>']
        self.eos = output_vocab['<eos>']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        # map OOV chars to <pad>=0 silently
        input_ids = [self.input_vocab.get(c, 0) for c in source]
        target_ids = [self.sos] + [self.output_vocab.get(c, 0) for c in target] + [self.eos]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def build_vocab(pairs):
    input_chars = set()
    output_chars = set()
    for source, target in pairs:
        input_chars.update(list(source))
        output_chars.update(list(target))
    # 0: <pad>
    input_vocab = {c: i + 1 for i, c in enumerate(sorted(input_chars))}
    input_vocab['<pad>'] = 0
    # 0:<pad> 1:<sos> 2:<eos>
    output_vocab = {c: i + 3 for i, c in enumerate(sorted(output_chars))}
    output_vocab.update({'<pad>': 0, '<sos>': 1, '<eos>': 2})
    return input_vocab, output_vocab

def invert_vocab(v):
    return {i: c for c, i in v.items()}

def load_pairs(path):
    # Dakshina TSV: target \t source \t count
    df = pd.read_csv(path, sep="\t", header=None, names=["target", "source", "count"], dtype=str)
    df.dropna(subset=["source", "target"], inplace=True)
    # Strip whitespace just in case
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    return list(zip(df["source"], df["target"]))

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    target_lens = [len(seq) for seq in targets]
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, input_lens, target_lens

# =============== Models ==================
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_class(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x, lengths):
        x = self.embedding(x)  # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        # We only need the final hidden(s) to initialize decoder
        return hidden  # GRU/RNN: (num_layers, B, H). LSTM: tuple((num_layers, B, H), (num_layers, B, H))

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        rnn_class = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_class(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden):
        """
        input_token: (B,) longs
        hidden: same type/shape as encoder hidden
        returns: logits (B, V), new_hidden
        """
        x = self.embedding(input_token).unsqueeze(1)  # (B, 1, E)
        output, hidden = self.rnn(x, hidden)          # output: (B, 1, H)
        logits = self.fc(output.squeeze(1))           # (B, V)
        return logits, hidden

    @torch.no_grad()
    def beam_search(self, hidden, max_len, sos_idx, eos_idx, beam_size=3):
        """
        Non-batched beam search (runs per sample). Handles LSTM/GRU hidden.
        hidden is either:
          - Tensor (num_layers, 1, H) OR
          - Tuple(h, c) with each (num_layers, 1, H)
        """
        device = next(self.parameters()).device

        def clone_hidden(h):
            if isinstance(h, tuple):
                return (h[0].clone(), h[1].clone())
            else:
                return h.clone()

        # Each item: (seq[LongTensor], hidden, log_prob)
        start_seq = torch.tensor([sos_idx], device=device, dtype=torch.long)
        sequences = [(start_seq, clone_hidden(hidden), 0.0)]
        completed = []

        for _ in range(max_len):
            new_sequences = []
            for seq, h, score in sequences:
                last_token = seq[-1].view(1)  # (1,)
                logits, new_h = self.forward(last_token, h)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
                topk_logp, topk_idx = torch.topk(log_probs, beam_size)

                for lp, idx in zip(topk_logp, topk_idx):
                    idx = idx.item()
                    new_seq = torch.cat([seq, torch.tensor([idx], device=device)])
                    new_score = score + lp.item()
                    new_sequences.append((new_seq, clone_hidden(new_h), new_score))

            # Keep top-k
            new_sequences.sort(key=lambda x: x[2], reverse=True)
            sequences = new_sequences[:beam_size]

            # Move completed to list
            still_running = []
            for seq, h, score in sequences:
                if seq[-1].item() == eos_idx:
                    completed.append((seq, h, score))
                else:
                    still_running.append((seq, h, score))
            sequences = still_running
            if not sequences:
                break

        if not completed:
            completed = sequences
        completed.sort(key=lambda x: x[2], reverse=True)
        return completed[0][0]  # best seq

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx=1, eos_idx=2, max_len=40):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

    def forward(self, src, src_lens, tgt=None, teacher_forcing_ratio=0.5):
        """
        If tgt is provided: training mode (returns logits over time).
        Else: returns list of token sequences (beam search per sample).
        """
        batch_size = src.size(0)
        device = src.device

        # Encode
        hidden = self.encoder(src, src_lens)

        if tgt is not None:
            tgt_len = tgt.size(1)
            vocab_size = self.decoder.fc.out_features
            outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=device)

            input_token = tgt[:, 0]  # <sos>
            dec_hidden = hidden

            for t in range(1, tgt_len):
                logits, dec_hidden = self.decoder(input_token, dec_hidden)
                outputs[:, t] = logits
                teacher_force = torch.rand(1, device=device).item() < teacher_forcing_ratio
                next_token = tgt[:, t] if teacher_force else torch.argmax(logits, dim=-1)
                input_token = next_token
            return outputs
        else:
            # Inference: beam search per example with batch_size=1 hidden slices
            sequences = []
            # Split hidden for each item in batch
            for b in range(batch_size):
                if isinstance(hidden, tuple):
                    h_b = tuple(h[:, b:b+1, :].contiguous() for h in hidden)
                else:
                    h_b = hidden[:, b:b+1, :].contiguous()
                seq = self.decoder.beam_search(
                    h_b, max_len=self.max_len, sos_idx=self.sos_idx, eos_idx=self.eos_idx, beam_size=3
                )
                sequences.append(seq)
            return sequences

# =============== Metrics =================
def char_accuracy(logits, targets, pad_idx=0):
    """
    logits: (B, T, V), targets: (B, T)
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = (targets != pad_idx)
        correct = ((preds == targets) & mask).sum().item()
        total = mask.sum().item()
        return (correct / total) if total > 0 else 0.0

def levenshtein(a, b):
    """
    Simple DP Levenshtein distance between two strings.
    """
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )
    return dp[n][m]

def decode_greedy(model, src, src_lens, output_ivocab, max_len=40, sos_idx=1, eos_idx=2):
    """
    Greedy decoding for batch (faster than beam for eval metrics).
    Returns list of strings.
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)

    # Encode
    hidden = model.encoder(src, src_lens)

    # Initialize
    input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
    dec_hidden = hidden
    outputs = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        logits, dec_hidden = model.decoder(input_token, dec_hidden)  # (B, V)
        next_token = torch.argmax(logits, dim=-1)                    # (B,)
        for b in range(batch_size):
            outputs[b].append(next_token[b].item())
        input_token = next_token

    # Convert ids to strings, stopping at eos
    decoded = []
    for seq in outputs:
        chars = []
        for tok in seq:
            if tok == eos_idx:
                break
            if tok in output_ivocab:
                ch = output_ivocab[tok]
                if ch not in ['<pad>', '<sos>', '<eos>']:
                    chars.append(ch)
        decoded.append(''.join(chars))
    return decoded

def batch_word_accuracy_and_cer(pred_strs, tgt_strs):
    """
    pred_strs, tgt_strs: lists of strings length B
    Returns (word_acc, cer)
    """
    assert len(pred_strs) == len(tgt_strs)
    exact = 0
    total_char_err = 0
    total_char = 0
    for p, t in zip(pred_strs, tgt_strs):
        if p == t:
            exact += 1
        dist = levenshtein(p, t)
        total_char_err += dist
        total_char += max(len(t), 1)
    word_acc = exact / len(pred_strs) if pred_strs else 0.0
    cer = total_char_err / total_char if total_char > 0 else 0.0
    return word_acc, cer

# =============== Train/Eval ==============
def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm=5.0, teacher_forcing_ratio=0.5):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio)  # (B, T, V)
        # shift to ignore first token (<sos> position 0)
        loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        acc = char_accuracy(logits[:, 1:], tgt[:, 1:])
        loss.backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc
    n = len(loader)
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, output_ivocab, eos_idx=2, sos_idx=1, max_len=40):
    model.eval()
    total_loss, total_char_acc = 0.0, 0.0
    all_pred, all_gold = [], []
    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Evaluating", leave=False):
        src, tgt = src.to(device), tgt.to(device)

        # Teacher forcing OFF for loss/char-acc to mimic inference distribution
        logits = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
        loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        acc = char_accuracy(logits[:, 1:], tgt[:, 1:])
        total_loss += loss.item()
        total_char_acc += acc

        # Word-level metrics via greedy decode
        batch_pred = decode_greedy(model, src, src_lens, output_ivocab, max_len=max_len, sos_idx=sos_idx, eos_idx=eos_idx)

        # Convert gold to string for the same batch
        gold_strs = []
        for seq in tgt.cpu().numpy():
            chars = []
            for tok in seq[1:]:  # skip <sos>
                if tok == eos_idx or tok == 0:
                    break
                ch = output_ivocab.get(int(tok), '')
                if ch not in ['<pad>', '<sos>', '<eos>']:
                    chars.append(ch)
            gold_strs.append(''.join(chars))

        all_pred.extend(batch_pred)
        all_gold.extend(gold_strs)

    n = len(loader)
    avg_loss = total_loss / n if n > 0 else 0.0
    avg_char_acc = total_char_acc / n if n > 0 else 0.0
    word_acc, cer = batch_word_accuracy_and_cer(all_pred, all_gold)
    return avg_loss, avg_char_acc, word_acc, cer

# =============== Main / W&B =============
def main():
    # This function is called by wandb.agent in a sweep
    config = wandb.config

    # Give each run a readable name
    run_name = f"cell:{config.cell_type}_emb:{config.embed_size}_hid:{config.hidden_size}_L:{config.num_layers}_bs:{config.batch_size}_lr:{config.lr}"
    if hasattr(wandb.run, "name") and (wandb.run.name is None or wandb.run.name == ""):
        wandb.run.name = run_name

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    train_pairs = load_pairs("/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
    dev_pairs   = load_pairs("/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")

    input_vocab, output_vocab = build_vocab(train_pairs)
    output_ivocab = invert_vocab(output_vocab)

    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    dev_dataset   = TransliterationDataset(dev_pairs,   input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_dataset,   batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Model ----
    encoder = Encoder(
        input_size=len(input_vocab),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        cell_type=config.cell_type,
        dropout=config.dropout
    )
    decoder = Decoder(
        output_size=len(output_vocab),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        cell_type=config.cell_type,
        dropout=config.dropout
    )
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        sos_idx=output_vocab['<sos>'],
        eos_idx=output_vocab['<eos>'],
        max_len=40
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_word_acc = -1.0

    for epoch in range(config.epochs):
        train_loss, train_char_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_norm=5.0,
            teacher_forcing_ratio=getattr(config, "teacher_forcing", 0.5)
        )
        val_loss, val_char_acc, val_word_acc, val_cer = evaluate(
            model, dev_loader, criterion, device, output_ivocab,
            eos_idx=output_vocab['<eos>'], sos_idx=output_vocab['<sos>'], max_len=40
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_char_accuracy": train_char_acc,
            "val_loss": val_loss,
            "val_char_accuracy": val_char_acc,
            "val_word_accuracy": val_word_acc,
            "val_CER": val_cer
        })

        # Save best on word accuracy
        if val_word_acc > best_val_word_acc:
            best_val_word_acc = val_word_acc
            save_path = os.path.join(wandb.run.dir, "best_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": dict(config),
                "input_vocab": input_vocab,
                "output_vocab": output_vocab
            }, save_path)
            wandb.log({"best_model_path": save_path, "best_val_word_accuracy": best_val_word_acc})

# =============== Entry ================
if __name__ == "__main__":
    # Define sweep if running this file directly.
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_word_accuracy", "goal": "maximize"},
        "parameters": {
            "embed_size":  {"values": [64, 128]},
            "hidden_size": {"values": [128, 256]},
            "num_layers":  {"values": [1, 2]},
            "cell_type":   {"values": ["GRU", "LSTM"]},
            "dropout":     {"values": [0.1, 0.2, 0.3]},
            "lr":          {"min": 1e-4, "max": 5e-3},
            "batch_size":  {"values": [32, 64]},
            "epochs":      {"values": [8]},                  
            "teacher_forcing": {"values": [0.5, 0.6, 0.7]},
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="Dakshina-Transliteration")

    def sweep_main():
        # Make sure a run is created and config is readable
        wandb.init(project="Dakshina-Transliteration")
        main()
        wandb.finish()

    # Launch N runs (adjust count as you like)
    wandb.agent(sweep_id, function=sweep_main, count=8)
