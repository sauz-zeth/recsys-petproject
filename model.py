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