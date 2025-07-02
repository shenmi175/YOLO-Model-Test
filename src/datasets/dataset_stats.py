"""Dataset statistics utilities."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List


import logging
from .xml_loader import Annotation, load_dataset, DatasetConsistencyError

def compute_stats(annotations: List[Annotation]) -> Dict[str, object]:
    """Compute simple statistics for a list of ``Annotation`` objects."""
    class_counts: Counter[str] = Counter()
    for ann in annotations:
        class_counts.update(b.label for b in ann.boxes)
    return {
        "num_images": len(annotations),
        "class_counts": dict(class_counts),
        "num_boxes": sum(class_counts.values()),
    }


def stats_from_dir(root_dir: str) -> Dict[str, object]:
    """Load annotations from ``root_dir`` and compute stats."""
    try:
        annotations = load_dataset(root_dir)
    except DatasetConsistencyError as exc:
        logging.debug("Dataset issue: %s", exc)
        annotations = exc.annotations

    return compute_stats(annotations)


try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover - matplotlib optional
    plt = None  # type: ignore


def plot_class_distribution(stats: Dict[str, object], save_path: str | None = None) -> None:
    """Plot class distribution using matplotlib if available."""
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    counts = stats.get("class_counts", {})
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_title("Class distribution")
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)



