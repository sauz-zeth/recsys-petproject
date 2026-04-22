import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SemanticSeqDataset(Dataset):
    def __init__(self, encoded_sessions: list[list[int]], seq_len: int = 10, pad_token: int = 0):
        self.encoded_sessions = encoded_sessions
        self.seq_len = seq_len
        self.pad_token = pad_token
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for session in self.encoded_sessions:
            if len(session) < self.seq_len:
                session = [self.pad_token] * (self.seq_len - len(session)) + session
            
            for i in range(len(session) - self.seq_len + 1):
                window = session[i : i + self.seq_len]
                x = window[:-1]
                y = window[-1]
                samples.append((x, y))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class GenerativeRecommender(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # batch_first=True позволяет подавать данные в формате (batch, seq, feature)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len - 1]
        embedded = self.embedding(x) 
        # out: [batch_size, seq_len - 1, hidden_dim]
        out, _ = self.gru(embedded)
        
        last_hidden = out[:, -1, :]
        
        logits = self.fc(last_hidden)
        return logits

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

    Заменяет GRU на стек трансформер-блоков с казуальной маской.
    Каждый токен может смотреть только на предыдущие позиции,
    что соответствует задаче предсказания следующего айтема.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        max_seq_len: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN — стабильнее при малых данных
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.dropout(self.item_emb(x) + self.pos_emb(positions))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        return self.fc(self.norm(h[:, -1, :]))


def train_model(model, dataloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")