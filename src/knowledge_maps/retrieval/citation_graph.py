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
    }


def expansion_bridge_ids(judgments: list[PrerequisiteJudgment]) -> list[str]:
    return sorted(
        judgment.candidate_id for judgment in judgments if judgment.relation in EXPANSION_RELATIONS
    )


def add_second_hop_discoveries(
    target_arxiv_id: str,
    direct_candidates: Mapping[str, CandidateDiscovery],
    references_by_bridge: Mapping[str, list[CitationEvidence]],
) -> dict[str, CandidateDiscovery]:
    paths_by_candidate = {
        candidate_id: list(discovery.paths) for candidate_id, discovery in direct_candidates.items()
    }

    for bridge_id, references in references_by_bridge.items():
        bridge_path = direct_candidates[bridge_id].paths[0]
        for reference in references:
            if reference.source_arxiv_id == target_arxiv_id:
                continue
            path = CitationPath(citations=[reference, *bridge_path.citations])
            paths_by_candidate.setdefault(reference.source_arxiv_id, []).append(path)

    return {
        candidate_id: CandidateDiscovery(
            depth=1 if candidate_id in direct_candidates else 2,
            paths=paths,
        )
        for candidate_id, paths in paths_by_candidate.items()
    }


def materialize_candidates(
    discoveries: Mapping[str, CandidateDiscovery],
    papers: Mapping[str, Paper],
    target_arxiv_id: str,
) -> list[PrerequisiteCandidate]:
    candidates = []
    for candidate_id, discovery in discoveries.items():
        bridge_ids = {
            citation.target_arxiv_id
            for path in discovery.paths
            for citation in path.citations[:-1]
            if citation.target_arxiv_id != target_arxiv_id
        }
        candidates.append(
            PrerequisiteCandidate(
                paper=papers[candidate_id],
                discovery=discovery,
                supporting_papers=[papers[bridge_id] for bridge_id in sorted(bridge_ids)],
            )
        )
    return candidates
