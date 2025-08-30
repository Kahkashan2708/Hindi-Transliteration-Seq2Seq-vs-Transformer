import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import csv
from collections import namedtuple

# Dataset + Collate Fn
# -----------------------------
class DakshinaDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, inp_vocab=None, tgt_vocab=None, build_vocab=False):
        self.pairs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.pairs.append((parts[0], parts[1]))

        if build_vocab:
            self.inp_vocab = self.build_vocab([p[0] for p in self.pairs])
            self.tgt_vocab = self.build_vocab([p[1] for p in self.pairs])
        else:
            self.inp_vocab = inp_vocab
            self.tgt_vocab = tgt_vocab

    def build_vocab(self, texts):
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        idx = 4
        for text in texts:
            for ch in text:
                if ch not in vocab:
                    vocab[ch] = idx
                    idx += 1
        return vocab

    def encode(self, text, vocab):
        return [vocab.get(ch, vocab["<unk>"]) for ch in text]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, tgt = self.pairs[idx]
        inp_ids = [self.inp_vocab["<sos>"]] + self.encode(inp, self.inp_vocab) + [self.inp_vocab["<eos>"]]
        tgt_ids = [self.tgt_vocab["<sos>"]] + self.encode(tgt, self.tgt_vocab) + [self.tgt_vocab["<eos>"]]
        return torch.tensor(inp_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    inps, tgts = zip(*batch)
    max_inp = max(len(x) for x in inps)
    max_tgt = max(len(x) for x in tgts)
    inp_pad = torch.zeros(len(batch), max_inp, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inps, tgts)):
        inp_pad[i, :len(inp)] = inp
        tgt_pad[i, :len(tgt)] = tgt
    return inp_pad, tgt_pad


# -----------------------------
# Transformer Model
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(self, inp_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding_inp = nn.Embedding(inp_vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = nn.Embedding(500, d_model)
        self.pos_decoder = nn.Embedding(500, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=src.device).unsqueeze(0)

        src_emb = self.embedding_inp(src) * math.sqrt(self.d_model) + self.pos_encoder(src_pos)
        tgt_emb = self.embedding_tgt(tgt) * math.sqrt(self.d_model) + self.pos_decoder(tgt_pos)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_test(model, loader, inp_vocab, tgt_vocab, device, show_samples=True, save_csv=False):
    model.eval()
    total, correct = 0, 0
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    inv_inp_vocab = {v: k for k, v in inp_vocab.items()}

    samples, preds_list = [], []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            output = model(src, tgt_inp)
            pred_tokens = output.argmax(-1)

            total += tgt_out.numel()
            correct += (pred_tokens == tgt_out).sum().item()

            for i in range(src.size(0)):
                inp_text = "".join(inv_inp_vocab.get(x.item(), "") for x in src[i] if x.item() > 3)
                pred_text = "".join(inv_tgt_vocab.get(x.item(), "") for x in pred_tokens[i] if x.item() > 3)
                truth_text = "".join(inv_tgt_vocab.get(x.item(), "") for x in tgt[i] if x.item() > 3)
                preds_list.append([inp_text, pred_text, truth_text])
                if len(samples) < 10:
                    samples.append((inp_text, pred_text, truth_text))

    acc = correct / total
    if show_samples:
        print(f"\n Test Accuracy: {acc*100:.2f}%")
        for inp, pred, truth in samples:
            print(f"{inp:15} | Pred: {pred:20} | Truth: {truth}")

    if save_csv:
        df = pd.DataFrame(preds_list, columns=["Input", "Prediction", "Truth"])
        df.to_csv("test_predictions.csv", index=False)
        print("\n Predictions saved to: test_predictions.csv")

    return acc


# -----------------------------
# Training Loop
# -----------------------------
def train_model(model, train_loader, test_loader, inp_vocab, tgt_vocab, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_inp)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} Train Loss: {avg_loss:.4f}")

        acc = evaluate_test(model, test_loader, inp_vocab, tgt_vocab, device)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_transformer.pt")

    print("\n Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_transformer.pt", map_location=device))
    final_acc = evaluate_test(model, test_loader, inp_vocab, tgt_vocab, device, save_csv=True)
    return final_acc


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = "/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    test_file = "/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"

    # Load datasets
    train_dataset = DakshinaDataset(train_file, build_vocab=True)
    test_dataset = DakshinaDataset(test_file, inp_vocab=train_dataset.inp_vocab, tgt_vocab=train_dataset.tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # Init model
    model = TransformerModel(len(train_dataset.inp_vocab), len(train_dataset.tgt_vocab)).to(device)

    # Train + Eval
    train_model(model, train_loader, test_loader, train_dataset.inp_vocab, train_dataset.tgt_vocab, device, epochs=10)


if __name__ == "__main__":
    main()
