from knowledge_maps.schemas import (
    CandidateFailure,
    CitationEvidence,
    ClassificationResult,
    ExpansionLevelMetrics,
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
        relations: dict[tuple[str, str], PrerequisiteRelation],
        failed_edges: set[tuple[str, str]] | None = None,
    ) -> None:
        self.relations = relations
        self.failed_edges = failed_edges or set()
        self.calls: list[list[PrerequisiteCandidate]] = []
        self.root_target_ids: list[str] = []
        self.immediate_target_ids: list[str] = []

    @property
    def name(self) -> str:
        return "test-model"

    def classify(
        self,
        root_target: Paper,
        immediate_target: Paper,
        candidates: list[PrerequisiteCandidate],
    ) -> ClassificationResult:
        assert all(candidate.target_id == immediate_target.arxiv_id for candidate in candidates)
        self.root_target_ids.append(root_target.arxiv_id)
        self.immediate_target_ids.append(immediate_target.arxiv_id)
        self.calls.append(candidates)
        return ClassificationResult(
            judgments=[
                PrerequisiteJudgment(
                    candidate_id=candidate.paper.arxiv_id,
                    target_id=candidate.target_id,
                    relation=self.relations[(candidate.paper.arxiv_id, candidate.target_id)],
                    evidence=f"Evidence for {candidate.paper.title}.",
                )
                for candidate in candidates
                if (candidate.paper.arxiv_id, candidate.target_id) not in self.failed_edges
            ],
            failures=[
                CandidateFailure(
                    candidate_id=candidate.paper.arxiv_id,
                    target_id=candidate.target_id,
                    retrieval_depth=candidate.discovery.depth,
                    attempts=3,
                    error="Model endpoint returned HTTP 503",
                )
                for candidate in candidates
                if (candidate.paper.arxiv_id, candidate.target_id) in self.failed_edges
            ],
            inference_requests=len(candidates),
            checkpoint_hits=0,
        )


def test_service_expands_essential_branches_until_they_end() -> None:
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
    deep_foundation = Paper(
        arxiv_id="1700.00001",
        title="Deep Foundation",
        authors=["Author"],
    )
    optional_leaf = Paper(
        arxiv_id="1600.00001",
        title="Optional Leaf",
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
        shared_ancestor.arxiv_id: [
            _citation(
                deep_foundation,
                shared_ancestor,
                "The shared ancestor depends on this foundation.",
            )
        ],
        deep_foundation.arxiv_id: [
            _citation(optional_leaf, deep_foundation, "This provides optional background."),
            _citation(target, deep_foundation, "This would create a cycle."),
        ],
    }
    reference_source = ReferenceSourceStub(references)
    model = ModelStub(
        {
            (foundation.arxiv_id, target.arxiv_id): PrerequisiteRelation.ESSENTIAL,
            (comparison.arxiv_id, target.arxiv_id): PrerequisiteRelation.RELATED_ONLY,
            (method.arxiv_id, target.arxiv_id): PrerequisiteRelation.ESSENTIAL,
            (shared_ancestor.arxiv_id, foundation.arxiv_id): PrerequisiteRelation.ESSENTIAL,
            (unrelated_ancestor.arxiv_id, foundation.arxiv_id): (PrerequisiteRelation.NOT_RELEVANT),
            (shared_ancestor.arxiv_id, method.arxiv_id): PrerequisiteRelation.ESSENTIAL,
            (foundation.arxiv_id, method.arxiv_id): PrerequisiteRelation.RELATED_ONLY,
            (deep_foundation.arxiv_id, shared_ancestor.arxiv_id): (PrerequisiteRelation.ESSENTIAL),
            (optional_leaf.arxiv_id, deep_foundation.arxiv_id): PrerequisiteRelation.HELPFUL,
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
                deep_foundation,
                optional_leaf,
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
        shared_ancestor.arxiv_id,
        deep_foundation.arxiv_id,
    ]
    assert [
        [candidate.paper.arxiv_id for candidate in candidates] for candidates in model.calls
    ] == [
        [foundation.arxiv_id, comparison.arxiv_id, method.arxiv_id],
        [shared_ancestor.arxiv_id, unrelated_ancestor.arxiv_id],
        [shared_ancestor.arxiv_id, foundation.arxiv_id],
        [deep_foundation.arxiv_id],
        [optional_leaf.arxiv_id],
    ]
    assert model.root_target_ids == [target.arxiv_id] * 5
    assert model.immediate_target_ids == [
        target.arxiv_id,
        foundation.arxiv_id,
        method.arxiv_id,
        shared_ancestor.arxiv_id,
        deep_foundation.arxiv_id,
    ]

    assert result.target_arxiv_id == target.arxiv_id
    assert [paper.arxiv_id for paper in result.papers] == [
        target.arxiv_id,
        foundation.arxiv_id,
        method.arxiv_id,
        shared_ancestor.arxiv_id,
        deep_foundation.arxiv_id,
        optional_leaf.arxiv_id,
    ]
    relationships = {
        (relationship.source_arxiv_id, relationship.target_arxiv_id): relationship
        for relationship in result.relationships
    }
    assert relationships[(foundation.arxiv_id, target.arxiv_id)].provenance.retrieval_depth == 1
    assert relationships[(method.arxiv_id, target.arxiv_id)].provenance.retrieval_depth == 1
    shared_to_foundation = relationships[(shared_ancestor.arxiv_id, foundation.arxiv_id)]
    shared_to_method = relationships[(shared_ancestor.arxiv_id, method.arxiv_id)]
    assert shared_to_foundation.provenance.retrieval_depth == 2
    assert shared_to_method.provenance.retrieval_depth == 2
    deep_to_shared = relationships[(deep_foundation.arxiv_id, shared_ancestor.arxiv_id)]
    leaf_to_deep = relationships[(optional_leaf.arxiv_id, deep_foundation.arxiv_id)]
    assert deep_to_shared.provenance.retrieval_depth == 3
    assert len(deep_to_shared.provenance.paths) == 2
    assert leaf_to_deep.provenance.retrieval_depth == 4
    first_path = shared_to_foundation.provenance.paths[0]
    assert [
        (citation.source_arxiv_id, citation.target_arxiv_id) for citation in first_path.citations
    ] == [
        (shared_ancestor.arxiv_id, foundation.arxiv_id),
        (foundation.arxiv_id, target.arxiv_id),
    ]
    assert shared_to_foundation.provenance.candidate_source == "semantic_scholar_citation_graph"
    assert result.generation.model == "test-model"
    assert result.generation.complete is True
    assert result.generation.failed_candidates == []
    assert result.generation.generated_at.tzinfo is not None
    assert result.generation.metrics.papers_expanded == 5
    assert result.generation.metrics.unique_candidates_considered == 7
    assert result.generation.metrics.inference_requests == 9
    assert result.generation.metrics.checkpoint_hits == 0
    assert result.generation.metrics.levels == [
        ExpansionLevelMetrics(
            depth=0,
            papers_expanded=1,
            candidates_classified=3,
            essential_edges=2,
            helpful_edges=0,
            failed_classifications=0,
        ),
        ExpansionLevelMetrics(
            depth=1,
            papers_expanded=2,
            candidates_classified=4,
            essential_edges=2,
            helpful_edges=0,
            failed_classifications=0,
        ),
        ExpansionLevelMetrics(
            depth=2,
            papers_expanded=1,
            candidates_classified=1,
            essential_edges=1,
            helpful_edges=0,
            failed_classifications=0,
        ),
        ExpansionLevelMetrics(
            depth=3,
            papers_expanded=1,
            candidates_classified=1,
            essential_edges=0,
            helpful_edges=1,
            failed_classifications=0,
        ),
    ]


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
            {(foundation.arxiv_id, target.arxiv_id): PrerequisiteRelation.ESSENTIAL},
            failed_edges={(failed.arxiv_id, target.arxiv_id)},
        ),
    )

    result = service.build(target.arxiv_id)

    assert result.generation.complete is False
    assert result.generation.failed_candidates == [
        CandidateFailure(
            candidate_id=failed.arxiv_id,
            target_id=target.arxiv_id,
            retrieval_depth=1,
            attempts=3,
            error="Model endpoint returned HTTP 503",
        )
    ]
    assert [relationship.source_arxiv_id for relationship in result.relationships] == [
        foundation.arxiv_id
    ]
    assert result.generation.metrics.levels[0].failed_classifications == 1


def _citation(source: Paper, target: Paper, context: str) -> CitationEvidence:
    return CitationEvidence(
        source_arxiv_id=source.arxiv_id,
        target_arxiv_id=target.arxiv_id,
        contexts=[context],
        intents=["background"],
        is_influential=False,
    )
