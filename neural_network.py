from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from data_handling import DataManager


class MultiClassClassifier(nn.Module):
    def __init__(self, dataset: DataManager = None, load_if_exists=True):
        super(MultiClassClassifier, self).__init__()
        if dataset is None:
            dataset = DataManager()
        self.dataset = dataset

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab, dim) for vocab, dim in dataset.emb_dims]
        )
        total = sum(dim for _, dim in dataset.emb_dims) + len(dataset.num_cols)

        self.fully_connected1 = nn.Linear(total, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fully_connected2 = nn.Linear(100, 300)
        self.batch_norm2 = nn.BatchNorm1d(300)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fully_connected3 = nn.Linear(300, 100)
        self.batch_norm3 = nn.BatchNorm1d(100)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.fully_connected4 = nn.Linear(100, len(dataset.labels))
        path = Path("model.tar")
        if load_if_exists and path.is_file():
            self.load(path)

    def forward(self, X_categorical, X_non_categorical):
        emb_outputs = [
            emb(X_categorical[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        x = torch.cat(emb_outputs, dim=1)
        x = torch.cat([x, X_non_categorical], dim=1)

        x = self.fully_connected1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fully_connected2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fully_connected3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fully_connected4(x)
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Loaded model")

    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Saved model")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save("model.tar")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manager = DataManager()
    x_cat, x_num = manager.get_train_features()
    y = manager.get_train_target()

    x_cat = x_cat.to(device)
    x_num = x_num.to(device)
    y = y.to(device)

    train_dataset = TensorDataset(x_cat, x_num, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    with MultiClassClassifier(dataset=manager) as model:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(50):
            epoch_loss = 0
            for cat_b, num_b, y_b in train_loader:
                optimizer.zero_grad()
                outputs = model(cat_b, num_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f"Average loss in epoch[{epoch}]: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            preds_train = torch.argmax(model(x_cat, x_num), dim=1)
            correct = (preds_train == y).sum().item()
            accuracy = correct / y.shape[0]

            print(f"Train Accuracy: {accuracy:.2%}")

            x_cat_test, x_num_test = manager.get_test_features()
            x_cat_test = x_cat_test.to(device)
            x_num_test = x_num_test.to(device)
            predictions = model(x_cat_test, x_num_test)
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
        np.savetxt("output.txt", preds, fmt="%d")


if __name__ == "__main__":
    main()
