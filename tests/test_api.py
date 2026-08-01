from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from knowledge_maps.api import create_app
from knowledge_maps.errors import (
    CheckpointError,
    ExternalServiceError,
    ModelOutputError,
    PaperNotFoundError,
)
from knowledge_maps.schemas import (
    ExpansionLevelMetrics,
    GenerationMetadata,
    GenerationMetrics,
    KnowledgeMap,
    Paper,
)


class SuccessfulServiceStub:
    def __init__(self, result: KnowledgeMap) -> None:
        self.result = result

    def build(self, _: str) -> KnowledgeMap:
        return self.result


class FailingServiceStub:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def build(self, _: str) -> KnowledgeMap:
        raise self.error


def test_endpoint_returns_the_graph_contract() -> None:
    graph = _knowledge_map()
    app = create_app(SuccessfulServiceStub(graph))  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/knowledge-maps", json={"arxiv_id_or_url": "2000.00001"})

    assert response.status_code == 200
    assert response.json() == graph.model_dump(mode="json")


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ValueError("Invalid arXiv ID"), 422),
        (PaperNotFoundError("Paper not found"), 404),
        (ExternalServiceError("Source unavailable"), 502),
        (ModelOutputError("Invalid model output"), 502),
        (CheckpointError("Checkpoint unavailable"), 500),
    ],
)
def test_endpoint_maps_expected_failures(error: Exception, expected_status: int) -> None:
    app = create_app(FailingServiceStub(error))  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post("/knowledge-maps", json={"arxiv_id_or_url": "2000.00001"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(error)}


def _knowledge_map() -> KnowledgeMap:
    return KnowledgeMap(
        target_arxiv_id="2000.00001",
        papers=[Paper(arxiv_id="2000.00001", title="Target", authors=[])],
        relationships=[],
        generation=GenerationMetadata(
            model="test-model",
            generated_at=datetime(2026, 7, 31, 12, tzinfo=UTC),
            metrics=GenerationMetrics(
                duration_seconds=0,
                papers_expanded=1,
                unique_candidates_considered=0,
                inference_requests=0,
                checkpoint_hits=0,
                levels=[
                    ExpansionLevelMetrics(
                        depth=0,
                        papers_expanded=1,
                        candidates_classified=0,
                        essential_edges=0,
                        helpful_edges=0,
                        failed_classifications=0,
                    )
                ],
            ),
        ),
    )
