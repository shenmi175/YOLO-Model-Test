"""Confusion matrix plotting utilities."""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:
    import matplotlib
    matplotlib.use("Agg")  # use a non-GUI backend to avoid warnings
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional deps
    plt = None  # type: ignore
    np = None  # type: ignore


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Iterable[str],
    normalize: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot a confusion matrix using matplotlib if available."""
    if plt is None or np is None:
        raise RuntimeError("matplotlib and numpy are required for plotting")

    arr = np.array(matrix, dtype=float)
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        arr = arr / row_sums

    fig, ax = plt.subplots()
    cax = ax.matshow(arr, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", va="center", ha="center")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)