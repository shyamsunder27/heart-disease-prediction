"""Thread-safe JSON file storage for the prototype (users and assessments)."""
import json
import os
import tempfile
import threading
from pathlib import Path

_lock = threading.RLock()


class JsonStore:
    def __init__(self, path, default):
        self.path = Path(path)
        self.default = default
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(default() if callable(default) else default)

    def _write(self, data):
        # Atomic write: write to a temp file, then replace.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def read(self):
        with _lock:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)

    def update(self, fn):
        """Run fn(data) -> result under a lock; data is saved afterwards."""
        with _lock:
            data = self.read()
            result = fn(data)
            self._write(data)
            return result
