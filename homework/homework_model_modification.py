"""Train linear and multiclass logistic regression models in PyTorch."""
from copy import deepcopy
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
SEED = 42


def prepare_outputs():
    """Create directories for the trained weights and figures."""
    for name in ("models", "plots"):
        (ROOT / name).mkdir(exist_ok=True)


def fit(train, validation, *, classes=None, epochs=150, batch_size=32,
        lr=0.03, optimizer_type=torch.optim.Adam, l1=0.0, l2=0.0, patience=None):
    """Train a linear layer and restore its best validation checkpoint."""
    torch.manual_seed(SEED)
    features, _ = train
    model = nn.Linear(features.shape[1], classes or 1)
    criterion = nn.CrossEntropyLoss() if classes else nn.MSELoss()
    optimizer = optimizer_type(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=True)
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss = loss + l1 * model.weight.abs().sum() + l2 * model.weight.square().sum()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss = criterion(model(validation[0]), validation[1]).item()
        history.append(loss)
        if epoch == 1 or epoch % 20 == 0:
            LOG.info("Epoch %d: validation loss %.4f", epoch, loss)
        if loss < best_loss:
            best_loss, stale = loss, 0
            best_weights = deepcopy(model.state_dict())
        else:
            stale += 1
            if patience is not None and stale >= patience:
                LOG.info("Early stopping at epoch %d", epoch)
                break
    model.load_state_dict(best_weights)
    return model, history


def tensors(x, y, *, classification=False):
    """Convert features and labels to the shapes expected by the loss."""
    features = torch.as_tensor(x, dtype=torch.float32)
    if classification:
        return features, torch.as_tensor(y, dtype=torch.long)
    return features, torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)


def split_data(x, y, *, classification=False):
    """Return reproducible train, validation and test splits (60/20/20)."""
    x_train, x_hold, y_train, y_hold = train_test_split(
        x, y, test_size=0.4, random_state=SEED, stratify=y if classification else None)
    x_val, x_test, y_val, y_test = train_test_split(
        x_hold, y_hold, test_size=0.5, random_state=SEED,
        stratify=y_hold if classification else None)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def classification_metrics(y, probabilities):
    """Compute macro metrics and probability-based ROC-AUC."""
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="macro", zero_division=0)
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(y, probabilities[:, 1])
    else:
        auc = roc_auc_score(y, probabilities, multi_class="ovr", average="macro")
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}


@torch.no_grad()
def evaluate(model, test, *, class_names=None, plot_name=None):
    """Report held-out MSE or classification metrics and save a confusion matrix."""
    x, y = test
    output = model(x)
    if class_names is None:
        result = {"mse": nn.functional.mse_loss(output, y).item()}
    else:
        probabilities = output.softmax(dim=1).numpy()
        result = classification_metrics(y.numpy(), probabilities)
        display = ConfusionMatrixDisplay.from_predictions(
            y.numpy(), probabilities.argmax(axis=1),
            labels=range(len(class_names)), display_labels=class_names)
        display.figure_.savefig(ROOT / "plots" / plot_name, bbox_inches="tight")
        plt.close(display.figure_)
    LOG.info("Test metrics: %s", {k: round(v, 4) for k, v in result.items()})
    return result


def main():
    """Compare regularized linear regression and train three-class logistic regression."""
    prepare_outputs()
    for name, dataset, classes in (("diabetes", load_diabetes(), None),
                                   ("iris", load_iris(), 3)):
        train, val, test = split_data(dataset.data, dataset.target, classification=bool(classes))
        scaler = StandardScaler().fit(train[0])
        train, val, test = [
            tensors(scaler.transform(x), y, classification=bool(classes))
            for x, y in (train, val, test)
        ]
        if classes is None:
            # Scale the target using training statistics for stable optimization.
            mean, std = train[1].mean(), train[1].std()
            train, val = [(x, (y - mean) / std) for x, y in (train, val)]
            plt.figure()
            for label, l1, l2 in (("baseline", 0, 0), ("L1", .01, 0),
                                  ("L2", 0, .01), ("L1 + L2", .01, .01)):
                model, history = fit(train, val, l1=l1, l2=l2, patience=15)
                with torch.no_grad():
                    mse = nn.functional.mse_loss(model(test[0]) * std + mean, test[1])
                LOG.info("%s %s: test MSE %.2f", name, label, mse.item())
                plt.plot(range(1, len(history) + 1), history, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Validation MSE (scaled target)")
            plt.legend()
            plt.savefig(ROOT / "plots" / "regularization.png", bbox_inches="tight")
            plt.close()
            checkpoint = {"state_dict": model.state_dict(), "target_mean": mean, "target_std": std}
        else:
            model, _ = fit(train, val, classes=classes, patience=15)
            evaluate(model, test, class_names=dataset.target_names, plot_name="iris_confusion.png")
            checkpoint = {"state_dict": model.state_dict(), "classes": dataset.target_names.tolist()}
        checkpoint.update(feature_mean=scaler.mean_.tolist(), feature_scale=scaler.scale_.tolist())
        torch.save(checkpoint, ROOT / "models" / f"{name}.pth")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
