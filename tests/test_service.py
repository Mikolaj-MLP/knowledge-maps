from knowledge_maps.schemas import (
    CandidateFailure,
    CitationEvidence,
    ClassificationResult,
    Paper,
    PrerequisiteCandidate,
    PrerequisiteJudgment,
    PrerequisiteRelation,
)
from knowledge_maps.service import KnowledgeMapService


class ArxivStub:
    def __init__(self, target: Paper, papers: list[Paper]) -> None:
        self.target = target
        self.papers = {paper.arxiv_id: paper for paper in papers}

    def get_paper(self, _: str) -> Paper:
        return self.target

    def get_papers(self, arxiv_ids: list[str]) -> list[Paper]:
        return [self.papers[arxiv_id] for arxiv_id in arxiv_ids]


class ReferenceSourceStub:
    def __init__(self, references: dict[str, list[CitationEvidence]]) -> None:
        self.references = references
        self.calls: list[str] = []

    def get_references(self, paper: Paper) -> list[CitationEvidence]:
        self.calls.append(paper.arxiv_id)
        return self.references[paper.arxiv_id]


class ModelStub:
    def __init__(
        self,
        relations: dict[str, PrerequisiteRelation],
        failed_ids: set[str] | None = None,
    ) -> None:
        self.relations = relations
        self.failed_ids = failed_ids or set()
        self.calls: list[list[PrerequisiteCandidate]] = []

    @property
    def name(self) -> str:
        return "test-model"

    def classify(
        self,
        _: Paper,
        candidates: list[PrerequisiteCandidate],
    ) -> ClassificationResult:
        self.calls.append(candidates)
        return ClassificationResult(
            judgments=[
                PrerequisiteJudgment(
                    candidate_id=candidate.paper.arxiv_id,
                    relation=self.relations[candidate.paper.arxiv_id],
                    evidence=f"Evidence for {candidate.paper.title}.",
                )
                for candidate in candidates
                if candidate.paper.arxiv_id not in self.failed_ids
            ],
            failures=[
                CandidateFailure(
                    candidate_id=candidate.paper.arxiv_id,
                    retrieval_depth=candidate.discovery.depth,
                    attempts=3,
                    error="Model endpoint returned HTTP 503",
                )
                for candidate in candidates
                if candidate.paper.arxiv_id in self.failed_ids
            ],
        )


def test_service_selectively_expands_and_classifies_two_hop_candidates() -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=["Researcher"])
    foundation = Paper(arxiv_id="1900.00001", title="Foundation", authors=["Author"])
    comparison = Paper(arxiv_id="1900.00002", title="Comparison", authors=["Author"])
    method = Paper(arxiv_id="1900.00003", title="Method", authors=["Author"])
    shared_ancestor = Paper(
        arxiv_id="1800.00001",
        title="Shared Ancestor",
        authors=["Author"],
    )
    unrelated_ancestor = Paper(
        arxiv_id="1800.00002",
        title="Unrelated Ancestor",
        authors=["Author"],
    )
    references = {
        target.arxiv_id: [
            _citation(foundation, target, "The target builds on this foundation."),
            _citation(comparison, target, "The target compares against this paper."),
            _citation(method, target, "The target uses this method."),
        ],
        foundation.arxiv_id: [
            _citation(shared_ancestor, foundation, "The foundation uses this result."),
            _citation(
                unrelated_ancestor,
                foundation,
                "The foundation compares against this paper.",
            ),
        ],
        method.arxiv_id: [
            _citation(shared_ancestor, method, "The method extends this result."),
            _citation(foundation, method, "The method also uses the foundation."),
        ],
    }
    reference_source = ReferenceSourceStub(references)
    model = ModelStub(
        {
            foundation.arxiv_id: PrerequisiteRelation.ESSENTIAL,
            comparison.arxiv_id: PrerequisiteRelation.RELATED_ONLY,
            method.arxiv_id: PrerequisiteRelation.HELPFUL,
            shared_ancestor.arxiv_id: PrerequisiteRelation.ESSENTIAL,
            unrelated_ancestor.arxiv_id: PrerequisiteRelation.NOT_RELEVANT,
        }
    )
    service = KnowledgeMapService(
        arxiv_client=ArxivStub(
            target,
            [
                foundation,
                comparison,
                method,
                shared_ancestor,
                unrelated_ancestor,
            ],
        ),  # type: ignore[arg-type]
        reference_client=reference_source,  # type: ignore[arg-type]
        prerequisite_model=model,
    )

    result = service.build(target.arxiv_id)

    assert reference_source.calls == [
        target.arxiv_id,
        foundation.arxiv_id,
        method.arxiv_id,
    ]
    assert [
        [candidate.paper.arxiv_id for candidate in candidates] for candidates in model.calls
    ] == [
        [foundation.arxiv_id, comparison.arxiv_id, method.arxiv_id],
        [shared_ancestor.arxiv_id, unrelated_ancestor.arxiv_id],
    ]
    second_hop_candidate = model.calls[1][0]
    assert [paper.arxiv_id for paper in second_hop_candidate.supporting_papers] == [
        foundation.arxiv_id,
        method.arxiv_id,
    ]

    assert result.target_arxiv_id == target.arxiv_id
    assert [paper.arxiv_id for paper in result.papers] == [
        target.arxiv_id,
        foundation.arxiv_id,
        method.arxiv_id,
        shared_ancestor.arxiv_id,
    ]
    relationships = {
        relationship.source_arxiv_id: relationship for relationship in result.relationships
    }
    assert relationships[foundation.arxiv_id].provenance.retrieval_depth == 1
    assert len(relationships[foundation.arxiv_id].provenance.paths) == 2
    assert relationships[shared_ancestor.arxiv_id].provenance.retrieval_depth == 2
    assert len(relationships[shared_ancestor.arxiv_id].provenance.paths) == 2
    first_path = relationships[shared_ancestor.arxiv_id].provenance.paths[0]
    assert [
        (citation.source_arxiv_id, citation.target_arxiv_id) for citation in first_path.citations
    ] == [
        (shared_ancestor.arxiv_id, foundation.arxiv_id),
        (foundation.arxiv_id, target.arxiv_id),
    ]
    assert (
        relationships[shared_ancestor.arxiv_id].provenance.candidate_source
        == "semantic_scholar_citation_graph"
    )
    assert result.generation.model == "test-model"
    assert result.generation.complete is True
    assert result.generation.failed_candidates == []
    assert result.generation.generated_at.tzinfo is not None


def test_service_returns_an_explicitly_incomplete_map_after_candidate_failure() -> None:
    target = Paper(arxiv_id="2000.00001", title="Target", authors=[])
    foundation = Paper(arxiv_id="1900.00001", title="Foundation", authors=[])
    failed = Paper(arxiv_id="1900.00002", title="Unavailable", authors=[])
    reference_source = ReferenceSourceStub(
        {
            target.arxiv_id: [
                _citation(foundation, target, "The target uses this method."),
                _citation(failed, target, "The target cites this paper."),
            ],
            foundation.arxiv_id: [],
        }
    )
    service = KnowledgeMapService(
        arxiv_client=ArxivStub(target, [foundation, failed]),  # type: ignore[arg-type]
        reference_client=reference_source,  # type: ignore[arg-type]
        prerequisite_model=ModelStub(
            {foundation.arxiv_id: PrerequisiteRelation.ESSENTIAL},
            failed_ids={failed.arxiv_id},
        ),
    )

    result = service.build(target.arxiv_id)

    assert result.generation.complete is False
    assert result.generation.failed_candidates == [
        CandidateFailure(
            candidate_id=failed.arxiv_id,
            retrieval_depth=1,
            attempts=3,
            error="Model endpoint returned HTTP 503",
        )
    ]
    assert [relationship.source_arxiv_id for relationship in result.relationships] == [
        foundation.arxiv_id
    ]


def _citation(source: Paper, target: Paper, context: str) -> CitationEvidence:
    return CitationEvidence(
        source_arxiv_id=source.arxiv_id,
        target_arxiv_id=target.arxiv_id,
        contexts=[context],
        intents=["background"],
        is_influential=False,
    )
