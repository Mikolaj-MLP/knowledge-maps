from datetime import UTC, datetime

from knowledge_maps.modeling.prerequisite import PrerequisiteModel
from knowledge_maps.output.graph import build_knowledge_map
from knowledge_maps.retrieval.citation_graph import (
    add_second_hop_discoveries,
    direct_discoveries,
    expansion_bridge_ids,
    materialize_candidates,
)
from knowledge_maps.schemas import GenerationMetadata, KnowledgeMap
from knowledge_maps.sources.arxiv import ArxivClient
from knowledge_maps.sources.semantic_scholar import SemanticScholarClient


class KnowledgeMapService:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        reference_client: SemanticScholarClient,
        prerequisite_model: PrerequisiteModel,
    ) -> None:
        self._arxiv_client = arxiv_client
        self._reference_client = reference_client
        self._prerequisite_model = prerequisite_model

    def build(self, arxiv_id_or_url: str) -> KnowledgeMap:
        target = self._arxiv_client.get_paper(arxiv_id_or_url)
        direct = direct_discoveries(self._reference_client.get_references(target))
        direct_papers = self._arxiv_client.get_papers(list(direct))
        papers_by_id = {paper.arxiv_id: paper for paper in direct_papers}
        direct_candidates = materialize_candidates(direct, papers_by_id, target.arxiv_id)
        direct_result = self._prerequisite_model.classify(target, direct_candidates)

        bridge_ids = expansion_bridge_ids(direct_result.judgments)
        references_by_bridge = {
            bridge_id: self._reference_client.get_references(papers_by_id[bridge_id])
            for bridge_id in bridge_ids
        }
        discoveries = add_second_hop_discoveries(
            target.arxiv_id,
            direct,
            references_by_bridge,
        )
        second_hop = {
            candidate_id: discovery
            for candidate_id, discovery in discoveries.items()
            if candidate_id not in direct
        }
        second_hop_papers = self._arxiv_client.get_papers(list(second_hop))
        papers_by_id.update({paper.arxiv_id: paper for paper in second_hop_papers})
        second_hop_candidates = materialize_candidates(
            second_hop,
            papers_by_id,
            target.arxiv_id,
        )
        second_hop_result = self._prerequisite_model.classify(
            target,
            second_hop_candidates,
        )

        candidates = materialize_candidates(discoveries, papers_by_id, target.arxiv_id)
        judgments = [*direct_result.judgments, *second_hop_result.judgments]
        failures = [*direct_result.failures, *second_hop_result.failures]
        generation = GenerationMetadata(
            model=self._prerequisite_model.name,
            generated_at=datetime.now(UTC),
            complete=not failures,
            failed_candidates=failures,
        )
        return build_knowledge_map(target, candidates, judgments, generation)
