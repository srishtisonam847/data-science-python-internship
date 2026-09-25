"""
Week 5 Deep Learning Application in Data Science
Project: Handwritten Digit Classification using a PyTorch Neural Network

Dataset:
    sklearn.datasets.load_digits
    1,797 samples, 64 input features (8x8 grayscale pixels), 10 classes.

Run:
    python week5_deep_learning_digits.py

Requirements:
    numpy, pandas, scikit-learn, matplotlib, torch
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = Path("week5_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class DigitMLP(nn.Module):
    """Fully connected neural network for 10-class digit classification."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.network(x)


def main():
    # 1. Load public dataset
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    # 2. Stratified 80/20 train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    # 3. Standardize using training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # 4. Create validation split from training set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=SEED,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=32,
        shuffle=True,
    )

    val_x = torch.tensor(X_val)
    val_y = torch.tensor(y_val)
    test_x = torch.tensor(X_test)

    # 5. Define model, loss and optimizer
    model = DigitMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # 6. Train with validation monitoring
    history = []
    best_val_loss = float("inf")
    best_state = None
    patience = 7
    wait = 0
    max_epochs = 50

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)
            total += len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_acc = (
                (val_logits.argmax(dim=1) == val_y)
                .float()
                .mean()
                .item()
            )

        history.append(
            (epoch, train_loss, train_acc, val_loss, val_acc)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    training_seconds = time.time() - start_time
    model.load_state_dict(best_state)

    # 7. Test evaluation
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        probabilities = torch.softmax(logits, dim=1).numpy()
        y_pred = logits.argmax(dim=1).numpy()

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n===== TEST RESULTS =====")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # 8. Baseline comparison
    baseline = LogisticRegression(max_iter=2000, random_state=SEED)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    print(f"Logistic-regression baseline accuracy: {baseline_acc:.4f}")
    print(
        "Neural-network improvement over baseline: "
        f"{(accuracy - baseline_acc) * 100:.2f} percentage points"
    )

    # 9. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    # 10. Save plots
    epochs = [h[0] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h[1] for h in history], label="Training loss")
    plt.plot(epochs, [h[3] for h in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h[2] for h in history], label="Training accuracy")
    plt.plot(epochs, [h[4] for h in history], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(
                j, i, int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    plt.colorbar(label="Number of samples")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # 11. Show misclassification examples in the console
    misclassified = np.where(y_pred != y_test)[0]
    print(f"\nMisclassified test samples: {len(misclassified)}")
    for idx in misclassified:
        confidence = probabilities[idx].max() * 100
        print(
            f"Index {idx}: true={y_test[idx]}, "
            f"predicted={y_pred[idx]}, confidence={confidence:.2f}%"
        )


if __name__ == "__main__":
    main()
