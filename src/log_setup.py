import logging
from datetime import datetime
from pathlib import Path


def setup_logging(path: str = "logs") -> logging.Logger:
    """Initialize logging to file and console.

    ``path`` can be either a directory or a full log file path. When a
    directory is provided, a timestamped file will be created inside it.
    """

    log_path = Path(path)
    if log_path.suffix:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_path / f"log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)