"""Logging utilities."""
import json
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "experiments") -> Path:
    """Set up logging directory and return the path."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = log_path / f"run_{timestamp}"
    run_path.mkdir(parents=True, exist_ok=True)
    
    return run_path


def log_metrics(metrics: dict, step: int, log_file: str = "metrics.jsonl"):
    """Append metrics to a JSONL file."""
    metrics["step"] = step
    metrics["timestamp"] = datetime.now().isoformat()
    
    with open(log_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def log_message(msg: str, level: str = "INFO"):
    """Print a formatted log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", file=sys.stderr)
