"""Parameters owned by local persistence."""

from pathlib import Path

# Checkpoints are generated runtime data and stay outside the source tree.
CHECKPOINT_DATABASE_PATH = Path(".knowledge_maps/checkpoints.sqlite3")
