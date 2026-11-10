"""Read CSV data, fit preprocessing on the training split and train both models."""
import csv
import logging

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

from homework_model_modification import ROOT, evaluate, fit, prepare_outputs, split_data, tensors


class CSVDataset(Dataset):
    """Store numeric, binary and categorical CSV features as tensors."""

    def __init__(self, features, targets, processor, *, classification=False):
        self.features, self.targets = tensors(
            processor.transform(features), targets, classification=classification)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def read_csv(path, target):
    """Read a header-based CSV with complete feature and target values."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or target not in reader.fieldnames:
            raise ValueError(f"Missing target column: {target}")
        columns = [name for name in reader.fieldnames if name != target]
        rows = list(reader)
    if not rows or not columns:
        raise ValueError("CSV must contain at least one row and one feature")
    if any(not row[name] for row in rows for name in [*columns, target]):
        raise ValueError("CSV contains missing values")
    return np.array([[row[name] for name in columns] for row in rows], dtype=object), np.array(
        [row[target] for row in rows])


def numeric_column(values):
    """Determine whether a complete training column can be parsed as numbers."""
    try:
        values.astype(float)
    except ValueError:
        return False
    return True


def fit_preprocessor(x):
    """Scale numeric columns and one-hot encode training categories."""
    numeric = [i for i in range(x.shape[1]) if numeric_column(x[:, i])]
    categorical = [i for i in range(x.shape[1]) if i not in numeric]
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ]).fit(x)


def write_csv(name, dataset):
    """Export bundled real-world datasets to CSV for offline reproduction."""
    path = ROOT / "data" / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([*dataset.feature_names, "target"])
            writer.writerows([*features, target] for features, target in zip(dataset.data, dataset.target))
    return path


def run_csv(path, *, classification=False):
    """Train and evaluate one model from a CSV file."""
    x, y = read_csv(path, "target")
    train, val, test = split_data(x, y if classification else y.astype(float),
                                  classification=classification)
    processor = fit_preprocessor(train[0])
    encoder = LabelEncoder().fit(train[1]) if classification else None
    datasets = [
        CSVDataset(x, encoder.transform(y) if classification else y, processor,
                   classification=classification)
        for x, y in (train, val, test)
    ]
    train, val, test = [(dataset.features, dataset.targets) for dataset in datasets]
    logging.info("Training on %s", path.name)
    if classification:
        model, _ = fit(train, val, classes=len(encoder.classes_), patience=15)
        evaluate(model, test, class_names=encoder.classes_, plot_name=f"{path.stem}_confusion.png")
        metadata = {"classes": encoder.classes_.tolist()}
    else:
        mean, std = train[1].mean(), train[1].std()
        train, val = [(x, (y - mean) / std) for x, y in (train, val)]
        model, _ = fit(train, val, patience=15)
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(model(test[0]) * std + mean, test[1])
        logging.info("Test MSE: %.2f", mse.item())
        metadata = {"target_mean": mean, "target_std": std}

    # Store fitted preprocessing alongside the model for later inference.
    import joblib
    joblib.dump(processor, ROOT / "models" / f"csv_{path.stem}_preprocessor.joblib")
    torch.save({"state_dict": model.state_dict(), **metadata},
               ROOT / "models" / f"csv_{path.stem}.pth")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare_outputs()
    run_csv(write_csv("diabetes", load_diabetes()))
    run_csv(write_csv("breast_cancer", load_breast_cancer()), classification=True)
