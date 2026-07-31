from pathlib import Path

from knowledge_maps.schemas import PrerequisiteJudgment, PrerequisiteRelation
from knowledge_maps.storage.checkpoints import JudgmentCheckpointStore


def test_checkpoint_store_persists_a_judgment_without_holding_the_database_open(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "checkpoints.sqlite3"
    store = JudgmentCheckpointStore(database_path)
    judgment = PrerequisiteJudgment(
        candidate_id="1900.00001",
        relation=PrerequisiteRelation.HELPFUL,
        evidence="It prepares the reader.",
    )

    store.save("fingerprint", "test-model", "2000.00001", judgment)

    assert store.get("fingerprint") == judgment
    database_path.unlink()
    assert database_path.exists() is False
