# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_path = os.path.join('data', 'asl_landmarks.csv')
model_path = os.path.join('models', 'asl_model.pt')

# ----------------------------
# 1. Dataset
# ----------------------------

class ASLDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        self.X = df.iloc[:, 1:].values.astype('float32')  # landmarks
        self.labels = df.iloc[:, 0].values

        # Encode labels: A-Z -> 0-25
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ----------------------------
# 2. Model
# ----------------------------

class ASLClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=25):
        super(ASLClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# 3. Training Loop
# ----------------------------

def train():
    dataset = ASLDataset(data_path)

    # Split dataset
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = ASLClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved as {model_path}")

if __name__ == '__main__':
    train()
