from knowledge_maps.output.config import INCLUDED_RELATIONS
from knowledge_maps.schemas import (
    GenerationMetadata,
    KnowledgeMap,
    Paper,
    PrerequisiteCandidate,
    PrerequisiteJudgment,
    PrerequisiteRelationship,
    RelationshipProvenance,
)


def build_knowledge_map(
    target: Paper,
    candidates: list[PrerequisiteCandidate],
    judgments: list[PrerequisiteJudgment],
    generation: GenerationMetadata,
) -> KnowledgeMap:
    candidates_by_edge = {
        (candidate.paper.arxiv_id, candidate.target_id): candidate for candidate in candidates
    }
    prerequisite_judgments = [
        judgment for judgment in judgments if judgment.relation in INCLUDED_RELATIONS
    ]
    prerequisite_papers = {
        judgment.candidate_id: candidates_by_edge[(judgment.candidate_id, judgment.target_id)].paper
        for judgment in prerequisite_judgments
    }
    relationships = [
        PrerequisiteRelationship(
            source_arxiv_id=judgment.candidate_id,
            target_arxiv_id=judgment.target_id,
            relation=judgment.relation,
            evidence=judgment.evidence,
            provenance=RelationshipProvenance(
                retrieval_depth=candidates_by_edge[
                    (judgment.candidate_id, judgment.target_id)
                ].discovery.depth,
                paths=candidates_by_edge[
                    (judgment.candidate_id, judgment.target_id)
                ].discovery.paths,
            ),
        )
        for judgment in prerequisite_judgments
    ]
    return KnowledgeMap(
        target_arxiv_id=target.arxiv_id,
        papers=[target, *prerequisite_papers.values()],
        relationships=relationships,
        generation=generation,
    )
