import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "server.log"
MAX_OLD_LOGS = 5


def _rotate_log_file():
    """Rename existing server.log to a timestamped backup, keeping at most MAX_OLD_LOGS."""
    if not LOG_FILE.exists():
        return
    mtime = datetime.fromtimestamp(LOG_FILE.stat().st_mtime)
    stamp = mtime.strftime("%Y-%m-%d_%H%M%S")
    LOG_FILE.rename(LOG_DIR / f"server.{stamp}.log")

    old_logs = sorted(LOG_DIR.glob("server.*.log"), key=lambda p: p.stat().st_mtime)
    for stale in old_logs[:-MAX_OLD_LOGS]:
        stale.unlink()


def setup_logging():
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_log_file()
    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger("gui.backend")
    root.setLevel(level)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)
    root.propagate = False

    if level <= logging.DEBUG:
        uvicorn_access = logging.getLogger("uvicorn.access")
        uvicorn_access.setLevel(logging.DEBUG)
