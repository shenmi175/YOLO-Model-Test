import logging
from datetime import datetime
from pathlib import Path


def setup_logging(path: str = "logs") -> logging.Logger:
    """Initialize run and debug log files.

    ``path`` can be either a directory or a full path to ``run.log``. If a
    directory is given, ``run.log`` and ``debug.log`` will be created inside it.
    ``debug.log`` only records errors while ``run.log`` captures general
    information.
    """

    log_path = Path(path)
    if log_path.suffix:  # a file path was provided
        log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log = log_path
    else:  # treat ``path`` as a directory
        log_path.mkdir(parents=True, exist_ok=True)
        run_log = log_path / "run.log"

    debug_log = run_log.with_name("debug.log")

    file_handler = logging.FileHandler(run_log)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    debug_handler = logging.FileHandler(debug_log)
    debug_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, console_handler, debug_handler],
    )
    return logging.getLogger(__name__)