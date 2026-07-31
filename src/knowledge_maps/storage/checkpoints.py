import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from knowledge_maps.errors import CheckpointError
from knowledge_maps.schemas import PrerequisiteJudgment


class JudgmentCheckpointStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise CheckpointError("Could not create the checkpoint directory") from error
        self._create_table()

    def get(self, fingerprint: str) -> PrerequisiteJudgment | None:
        try:
            with closing(sqlite3.connect(self._database_path)) as connection:
                row = connection.execute(
                    "SELECT judgment FROM judgments WHERE fingerprint = ?",
                    (fingerprint,),
                ).fetchone()
        except sqlite3.Error as error:
            raise CheckpointError("Could not read the judgment checkpoint") from error

        if row is None:
            return None
        try:
            return PrerequisiteJudgment.model_validate_json(row[0])
        except ValidationError as error:
            raise CheckpointError("Saved judgment checkpoint is invalid") from error

    def save(
        self,
        fingerprint: str,
        model: str,
        target_id: str,
        judgment: PrerequisiteJudgment,
    ) -> None:
        try:
            with closing(sqlite3.connect(self._database_path)) as connection, connection:
                connection.execute(
                    """
                        INSERT OR REPLACE INTO judgments (
                            fingerprint, model, target_id, candidate_id, judgment, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                    (
                        fingerprint,
                        model,
                        target_id,
                        judgment.candidate_id,
                        judgment.model_dump_json(),
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except sqlite3.Error as error:
            raise CheckpointError("Could not save the judgment checkpoint") from error

    def _create_table(self) -> None:
        try:
            with closing(sqlite3.connect(self._database_path)) as connection, connection:
                connection.execute(
                    """
                        CREATE TABLE IF NOT EXISTS judgments (
                            fingerprint TEXT PRIMARY KEY,
                            model TEXT NOT NULL,
                            target_id TEXT NOT NULL,
                            candidate_id TEXT NOT NULL,
                            judgment TEXT NOT NULL,
                            created_at TEXT NOT NULL
                        )
                        """
                )
        except sqlite3.Error as error:
            raise CheckpointError("Could not initialize judgment checkpoints") from error
