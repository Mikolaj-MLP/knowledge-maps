import json
from datetime import UTC, datetime

from knowledge_maps import cli
from knowledge_maps.schemas import (
    ExpansionLevelMetrics,
    GenerationMetadata,
    GenerationMetrics,
    KnowledgeMap,
    Paper,
)


class ServiceStub:
    def __init__(self, result: KnowledgeMap) -> None:
        self.result = result

    def build(self, _: str) -> KnowledgeMap:
        return self.result


def test_build_command_prints_graph_json(monkeypatch, capsys) -> None:
    graph = KnowledgeMap(
        target_arxiv_id="2000.00001",
        papers=[Paper(arxiv_id="2000.00001", title="Target with a ﬂ ligature", authors=[])],
        relationships=[],
        generation=GenerationMetadata(
            model="test-model",
            generated_at=datetime(2026, 7, 30, 12, tzinfo=UTC),
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
    monkeypatch.setattr(cli, "create_service", lambda: ServiceStub(graph))

    exit_code = cli.main(["build", "2000.00001"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(output.out) == graph.model_dump(mode="json")
    assert "ﬂ" not in output.out
    assert "\\ufb02" in output.out
    assert output.err == ""
