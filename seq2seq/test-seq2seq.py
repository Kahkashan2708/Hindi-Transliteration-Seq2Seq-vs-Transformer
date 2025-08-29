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
        input_ids = [self.input_vocab[c] for c in source]
        target_ids = [self.sos] + [self.output_vocab[c] for c in target] + [self.eos]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def load_pairs(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['target', 'source', 'count'], dtype=str)
    df.dropna(subset=["source", "target"], inplace=True)
    return list(zip(df['source'], df['target']))

def build_vocab(pairs):
    input_chars = set()
    output_chars = set()
    for src, tgt in pairs:
        input_chars.update(src)
        output_chars.update(tgt)
    input_vocab = {c: i+1 for i, c in enumerate(sorted(input_chars))}
    input_vocab['<pad>'] = 0
    output_vocab = {c: i+3 for i, c in enumerate(sorted(output_chars))}
    output_vocab.update({'<pad>': 0, '<sos>': 1, '<eos>': 2})
    return input_vocab, output_vocab

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(x) for x in inputs]
    target_lens = [len(x) for x in targets]
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded, input_lens, target_lens

# ---------------- Models ----------------
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell_type]
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, token, hidden):
        x = self.embedding(token.unsqueeze(1))
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, tgt=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        hidden = self.encoder(src, src_lens)
        tgt_len = tgt.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc.out_features).to(src.device)
        input_token = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else output.argmax(1)
        return outputs

# ---------------- Train + Eval ----------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt, src_lens, _ in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_lens, tgt)
        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_and_save(model, dataloader, input_vocab, output_vocab, device, csv_path=None):
    model.eval()
    inv_input_vocab = {v: k for k, v in input_vocab.items()}
    inv_output_vocab = {v: k for k, v in output_vocab.items()}
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for src, tgt, src_lens, _ in dataloader:
            src = src.to(device)
            hidden = model.encoder(src, src_lens)
            input_token = torch.tensor([output_vocab['<sos>']] * src.size(0)).to(device)
            decoded = []
            for _ in range(20):
                output, hidden = model.decoder(input_token, hidden)
                input_token = output.argmax(1)
                decoded.append(input_token)
            decoded = torch.stack(decoded, dim=1)

            for i in range(src.size(0)):
                pred = ''.join([inv_output_vocab[t.item()] for t in decoded[i] if t.item() not in [output_vocab['<eos>'], 0]])
                truth = ''.join([inv_output_vocab[t.item()] for t in tgt[i][1:-1]])
                inp = ''.join([inv_input_vocab[t.item()] for t in src[i] if t.item() != 0])
                results.append((inp, pred, truth))
                if pred == truth:
                    correct += 1
                total += 1

    acc = correct / total * 100
    print(f"\n Test Accuracy: {acc:.2f}%")
    for inp, pred, truth in results[:10]:
        print(f"{inp:<15} | Pred: {pred:<20} | Truth: {truth}")

    if csv_path is not None:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Input', 'Prediction', 'GroundTruth'])
            writer.writerows(results)
        print(f"\n Predictions saved to: {csv_path}")

    return acc, results


# ------------ Run ----------------
if __name__ == "__main__":
    config = {
        "embed_size": 128,
        "hidden_size": 256,
        "num_layers": 3,
        "cell_type": "LSTM",
        "dropout": 0.2,
        "batch_size": 32,
        "lr": 0.00082004,
        "epochs": 10,
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pairs = load_pairs("/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
    test_pairs = load_pairs("/kaggle/input/dakshina/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")
    input_vocab, output_vocab = build_vocab(train_pairs)
    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    test_dataset = TransliterationDataset(test_pairs, input_vocab, output_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(input_vocab), config["embed_size"], config["hidden_size"],
                      config["num_layers"], config["cell_type"], config["dropout"])
    decoder = Decoder(len(output_vocab), config["embed_size"], config["hidden_size"],
                      config["num_layers"], config["cell_type"], config["dropout"])
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_acc = 0
    for epoch in range(config["epochs"]):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        acc, results = evaluate_and_save(model, test_loader, input_vocab, output_vocab, device, csv_path=None)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

    print("\n Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth"))

    # Save predictions CSV here
    evaluate_and_save(model, test_loader, input_vocab, output_vocab, device, csv_path="test_predictions.csv")
