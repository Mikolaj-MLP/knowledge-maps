from collections.abc import Mapping

from knowledge_maps.retrieval.config import EXPANSION_RELATIONS
from knowledge_maps.schemas import (
    CandidateDiscovery,
    CitationEvidence,
    CitationPath,
    Paper,
    PrerequisiteCandidate,
    PrerequisiteJudgment,
)


def direct_discoveries(
    references: list[CitationEvidence],
) -> dict[str, CandidateDiscovery]:
    return {
        reference.source_arxiv_id: CandidateDiscovery(
            depth=1,
            paths=[CitationPath(citations=[reference])],
        )
        for reference in references
        if reference.source_arxiv_id != reference.target_arxiv_id
    }


def expansion_bridge_ids(judgments: list[PrerequisiteJudgment]) -> list[str]:
    return sorted(
        judgment.candidate_id for judgment in judgments if judgment.relation in EXPANSION_RELATIONS
    )


def next_hop_discoveries(
    parent_discovery: CandidateDiscovery,
    references: list[CitationEvidence],
) -> dict[str, CandidateDiscovery]:
    paths_by_candidate: dict[str, list[CitationPath]] = {}
    for reference in references:
        paths = [
            CitationPath(citations=[reference, *path.citations])
            for path in parent_discovery.paths
            if reference.source_arxiv_id not in _path_paper_ids(path)
        ]
        paths_by_candidate.setdefault(reference.source_arxiv_id, []).extend(paths)

    return {
        candidate_id: CandidateDiscovery(
            depth=parent_discovery.depth + 1,
            paths=_unique_paths(paths),
        )
        for candidate_id, paths in paths_by_candidate.items()
        if paths
    }


def merge_discoveries(
    first: CandidateDiscovery,
    second: CandidateDiscovery,
) -> CandidateDiscovery:
    return CandidateDiscovery(
        depth=min(first.depth, second.depth),
        paths=_unique_paths([*first.paths, *second.paths]),
    )


def materialize_candidates(
    discoveries: Mapping[str, CandidateDiscovery],
    papers: Mapping[str, Paper],
    target_id: str,
) -> list[PrerequisiteCandidate]:
    return [
        PrerequisiteCandidate(
            paper=papers[candidate_id],
            target_id=target_id,
            discovery=discovery,
        )
        for candidate_id, discovery in discoveries.items()
    ]


def _path_paper_ids(path: CitationPath) -> set[str]:
    return {
        *(citation.source_arxiv_id for citation in path.citations),
        path.citations[-1].target_arxiv_id,
    }


def _unique_paths(paths: list[CitationPath]) -> list[CitationPath]:
    unique: dict[tuple[tuple[str, str], ...], CitationPath] = {}
    for path in paths:
        key = tuple(
            (citation.source_arxiv_id, citation.target_arxiv_id) for citation in path.citations
        )
        unique[key] = path
    return list(unique.values())
