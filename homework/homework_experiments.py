"""Compare optimizer settings and feature engineering on the same data split."""
import csv
from itertools import product
import logging

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import torch

from homework_model_modification import ROOT, fit, plt, prepare_outputs, split_data, tensors


def feature_engineering(x, polynomial):
    """Add squares, pairwise interactions, row means and row variances."""
    return np.column_stack((polynomial.transform(x), x.mean(axis=1), x.var(axis=1)))


def prepare_splits(splits, *, engineered=False):
    """Fit feature transforms and target scaling only on the training set."""
    scaler = StandardScaler().fit(splits[0][0])
    xs = [scaler.transform(x) for x, _ in splits]
    if engineered:
        polynomial = PolynomialFeatures(degree=2, include_bias=False).fit(xs[0])
        xs = [feature_engineering(x, polynomial) for x in xs]
        final_scaler = StandardScaler().fit(xs[0])
        xs = [final_scaler.transform(x) for x in xs]
    train_targets = splits[0][1]
    mean, std = train_targets.mean(), train_targets.std()
    train, val, test = [tensors(x, (y - mean) / std) for x, (_, y) in zip(xs, splits)]
    return train, val, test, std


def score(model, data, scale):
    """Return MSE in the original target units."""
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(data[0]), data[1]).item() * scale**2


def main():
    """Select settings on validation data and compare features on held-out data."""
    prepare_outputs()
    data = load_diabetes()
    splits = split_data(data.data, data.target)
    train, val, test, scale = prepare_splits(splits)
    optimizers = {cls.__name__: cls for cls in (torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop)}
    rows = []
    for name, lr, batch in product(optimizers, (.001, .01, .05), (16, 32, 64)):
        model, _ = fit(train, val, epochs=60, batch_size=batch, lr=lr,
                       optimizer_type=optimizers[name])
        val_mse = score(model, val, scale)
        rows.append((name, lr, batch, val_mse))
        logging.info("%s lr=%g batch=%d: validation MSE %.2f", name, lr, batch, val_mse)

    rows.sort(key=lambda row: row[-1])
    with (ROOT / "plots" / "hyperparameters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["optimizer", "learning_rate", "batch_size", "validation_mse"])
        writer.writerows(rows)

    name, lr, batch, _ = rows[0]
    settings = dict(epochs=60, batch_size=batch, lr=lr, optimizer_type=optimizers[name])
    logging.info("Selected: %s lr=%g batch=%d", name, lr, batch)
    baseline, _ = fit(train, val, **settings)
    train, val, enhanced_test, scale = prepare_splits(splits, engineered=True)
    enhanced, _ = fit(train, val, **settings)
    scores = [score(baseline, test, scale), score(enhanced, enhanced_test, scale)]
    logging.info("Test MSE: baseline %.2f, engineered %.2f", *scores)
    fig, ax = plt.subplots()
    bars = ax.bar(["Baseline", "Engineered"], scores)
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("Test MSE (lower is better)")
    ax.set_title("Effect of feature engineering")
    fig.savefig(ROOT / "plots" / "feature_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
