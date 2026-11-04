"""
Neural Network Models for EE5130 MNIST Project using PyTorch (CPU-optimized).

Models:
- MLP (Multilayer Perceptron: 784 -> 256 -> 128 -> 10 with Dropout)
- CNN (Convolutional Neural Network: Conv32 -> Pool -> Conv64 -> Pool -> Dense128 -> Dense10)
- Autoencoder (784 -> 256 -> 64 -> 32 bottleneck -> 64 -> 256 -> 784)
All models expose uniform MNISTBaseClassifier interface.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.classifiers import MNISTBaseClassifier
from src.utils import MODELS_DIR, set_seed


class MLPArchitecture(nn.Module):
    def __init__(self, input_dim: int = 784, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class CNNArchitecture(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        feats = self.features(x)
        return self.classifier(feats)


class AutoencoderArchitecture(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


class PyTorchClassifierWrapper(MNISTBaseClassifier):
    """Unified wrapper connecting PyTorch neural network to standard model zoo interface."""

    def __init__(self, net: nn.Module, name: str, is_cnn: bool = False):
        super().__init__(name=name, family="Deep Learning")
        self.net = net
        self.is_cnn = is_cnn
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 4,
    ) -> "PyTorchClassifierWrapper":
        set_seed(42)
        device = torch.device("cpu")
        self.net.to(device)

        X_tr = torch.tensor(X_train.astype(np.float32) / 255.0, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        if self.is_cnn:
            X_tr = X_tr.view(-1, 1, 28, 28)

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val.astype(np.float32) / 255.0, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.long)
            if self.is_cnn:
                X_v = X_v.view(-1, 1, 28, 28)
            val_ds = TensorDataset(X_v, y_v)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_weights = None
        no_improve_epochs = 0

        for epoch in range(epochs):
            self.net.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

            train_loss = total_loss / total
            train_acc = correct / total
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation
            if val_loader is not None:
                self.net.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        logits = self.net(batch_x)
                        loss = criterion(logits, batch_y)
                        v_loss += loss.item() * len(batch_y)
                        preds = torch.argmax(logits, dim=1)
                        v_correct += (preds == batch_y).sum().item()
                        v_total += len(batch_y)

                val_loss = v_loss / v_total
                val_acc = v_correct / v_total
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                print(f"  Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.clone() for k, v in self.net.state_dict().items()}
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"  Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                print(f"  Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%")

        if best_weights is not None:
            self.net.load_state_dict(best_weights)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, f"{self.name} is not fitted."
        self.net.eval()
        arr = (X.astype(np.float32) / 255.0) if X.max() > 1.0 else X.astype(np.float32)
        tensor_x = torch.tensor(arr, dtype=torch.float32)
        if self.is_cnn and tensor_x.dim() == 2:
            tensor_x = tensor_x.view(-1, 1, 28, 28)

        loader = DataLoader(TensorDataset(tensor_x), batch_size=256, shuffle=False)
        probas = []
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = self.net(batch_x)
                probs = torch.softmax(logits, dim=1).numpy()
                probas.append(probs)
        return np.vstack(probas)


def train_autoencoder(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    latent_dim: int = 32,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[AutoencoderArchitecture, Dict[str, List[float]]]:
    """Train Autoencoder on training set to obtain compact latent space."""
    set_seed(42)
    device = torch.device("cpu")
    ae = AutoencoderArchitecture(latent_dim=latent_dim).to(device)

    X_tr = torch.tensor(X_train.astype(np.float32) / 255.0, dtype=torch.float32)
    train_ds = TensorDataset(X_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    print(f"\nTraining Autoencoder (Bottleneck: {latent_dim}-d)...")
    for epoch in range(epochs):
        ae.train()
        total_loss = 0.0
        total = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = ae(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            total += len(batch_x)

        epoch_loss = total_loss / total
        history["train_loss"].append(epoch_loss)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:2d}/{epochs}: Reconstruction MSE Loss = {epoch_loss:.5f}")

    return ae, history

