import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def test():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Device:", device)
    
    n_features = 10
    look_back = 4
    lstm_units = 64
    dropout_rate = 0.2

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout_r):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(dropout_r)
            self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
            self.fc1 = nn.Linear(hidden_dim // 2, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.dropout(out)
            out, _ = self.lstm2(out)
            out = out[:, -1, :] # Keep only last time step
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = SimpleLSTM(n_features, lstm_units, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_tr_seq = np.random.rand(25, 4, 10).astype(np.float32)
    y_tr_seq = np.random.rand(25).astype(np.float32)

    X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).view(-1, 1).to(device)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_tr_t)
        loss = criterion(outputs, y_tr_t)
        loss.backward()
        optimizer.step()
        print("Epoch", epoch, loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_tr_t).cpu().numpy().ravel()
        print("Preds shape:", preds.shape)

if __name__ == "__main__":
    test()
